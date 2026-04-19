#!/usr/bin/env python3
"""Quantize activations (int8wa): weights AND activations INT8.

This matches ORT's Nano-int8 shape exactly — no fp16 activation path
remains at inference, so CoreML runs on CPU at INT8 matmul like ORT.

Generates a small calibration set per bucket (matching that bucket's
fixed input shape), feeds it to
`coremltools.optimize.coreml.linear_quantize_activations`, and saves
the activation-quantized .mlpackage to scripts/models/int8wa/.

Calibration samples:
- TextStage: random realistic phoneme IDs (1..170) of shape (1, L),
  random style row from voices.npz, mask with random "real" prefix.
- GeneratorStage: random normal tensors matching the actual output
  distribution of the fp32 TextStage → length regulation pipeline.
  We compute (mean, std) from 1 real text input once.

Usage:
    python scripts/calibrate_and_quantize.py text      -L 16,32,64,128,400
    python scripts/calibrate_and_quantize.py generator -N 128,256,512,1024
    python scripts/calibrate_and_quantize.py both
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto

# coremltools' linear_quantize_activations emits fp16 scale values, so the
# source model must already be fp16 — an fp32 source fails the validator
# with "scale has dtype fp16 whereas input has dtype fp32".
SRC_DIR = Path("scripts/models/fp16")
DST_DIR = Path("scripts/models/int8wa")
VOICES_NPZ = Path("scripts/models/voices.npz")
# Real (prosody_lr, text_lr, style) tuples produced by real_calibration.py.
# Each N-bucket subdir holds one .npz per sample.
CALIB_TUPLES_DIR = Path("scripts/models/calibration_tuples")

N_CALIB = 32


def _voices_sample(k: int, rng: np.random.Generator) -> np.ndarray:
    """Return k random style vectors (shape (1, 256) each)."""
    v = np.load(str(VOICES_NPZ))
    # Pool all 8 voice tables together — ~3200 rows.
    rows = np.concatenate([v[name] for name in v.files], axis=0).astype(np.float32)
    idx = rng.integers(0, rows.shape[0], size=k)
    return rows[idx].reshape(k, 1, 256)


def _text_calibration(L: int, n: int, rng: np.random.Generator) -> list[dict]:
    """Load real phonemized (input_ids, style, attention_mask) tuples.
    Falls back to random IDs if real_calibration.py hasn't been run yet —
    random phonemes are structurally similar enough to train *weights*
    but not *duration* heads, so prefer real samples when available.
    """
    bucket_dir = CALIB_TUPLES_DIR / f"L{L}"
    if bucket_dir.exists() and any(bucket_dir.glob("sample_*.npz")):
        paths = sorted(bucket_dir.glob("sample_*.npz"))
        chosen = paths[: n]
        samples: list[dict] = []
        for p in chosen:
            z = np.load(p)
            samples.append({
                "input_ids":      z["input_ids"].astype(np.int32),
                "style":          z["style"].reshape(1, 256).astype(np.float32),
                "attention_mask": z["attention_mask"].astype(np.float32),
            })
        return samples

    # Fallback (legacy path). Used only if real_calibration.py absent.
    samples = []
    styles = _voices_sample(n, rng)
    for i in range(n):
        real_L = max(4, int(rng.integers(max(4, L // 4), L + 1)))
        ids = np.zeros((1, L), dtype=np.int32)
        ids[0, :real_L] = rng.integers(1, 170, size=real_L, dtype=np.int32)
        mask = np.zeros((1, L), dtype=np.float32)
        mask[0, :real_L] = 1.0
        samples.append({
            "input_ids":      ids,
            "style":          styles[i],
            "attention_mask": mask,
        })
    return samples


def _generator_calibration(N: int, n: int, rng: np.random.Generator) -> list[dict]:
    """Load real length-regulated tuples produced by real_calibration.py.

    Gaussians proved insufficient: the generator's activation stats depend
    strongly on the repeated-frame / zero-tail structure that length
    regulation creates, not just marginal mean/std. Bad calibration = bad
    scales = distorted audio (slow robotic / cave echo / whisper).

    `n` is a ceiling; we use whatever real tuples exist for this bucket.
    """
    bucket_dir = CALIB_TUPLES_DIR / f"N{N}"
    if not bucket_dir.exists():
        raise SystemExit(
            f"missing {bucket_dir}; run real_calibration.py first")
    paths = sorted(bucket_dir.glob("sample_*.npz"))
    if not paths:
        raise SystemExit(
            f"no real calibration samples in {bucket_dir}")
    chosen = paths[: n] if len(paths) > n else paths
    samples: list[dict] = []
    for p in chosen:
        z = np.load(p)
        samples.append({
            "prosody_lr": z["prosody_lr"].astype(np.float32),
            "text_lr":    z["text_lr"].astype(np.float32),
            "style":      z["style"].reshape(1, 256).astype(np.float32),
        })
    # Pad with rng-picked repeats if we have fewer than n samples (keeps
    # downstream code's expectation of at least a handful; repetition is
    # harmless — the quantizer already saw the same stats once).
    if len(samples) < 4 and rng is not None:  # cheap minimum
        while len(samples) < 4:
            samples.append(samples[rng.integers(0, len(samples))])
    return samples


def quantize_one(src: Path, dst: Path, calib: list[dict]) -> tuple[int, int]:
    if dst.exists():
        shutil.rmtree(dst)
    print(f"  loading {src.name}")
    model = ct.models.MLModel(str(src))

    # Step 1: quantize activations with calibration.
    # Attempt #2 workaround: whitelist only the ops where activation
    # quantization is well-defined. `gather` (embedding lookup) takes int32
    # indices — coremltools tries to insert an fp32 scale on that int32
    # input and hits a dtype validator. Deprecated `op_selector` is the
    # documented filter but is disabled in ct 9. Using per-op-type config
    # with `global_config=None` restricts quantization to the named types.
    linear_cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric")
    act_cfg = cto.OptimizationConfig(
        global_config=None,
        op_type_configs={
            "linear": linear_cfg,
            "matmul": linear_cfg,
            "conv":   linear_cfg,
            # Omit lstm/gather/softmax/etc — they either can't be quantized
            # cleanly or break the validator.
        },
    )
    print(f"    step 1/2: activation calibration on {len(calib)} samples ...")
    model_a8 = cto.linear_quantize_activations(
        model, act_cfg, sample_data=calib,
    )

    # Step 2: quantize weights on top, producing w8a8.
    w_cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"
        )
    )
    print("    step 2/2: weight quantization ...")
    model_w8a8 = cto.linear_quantize_weights(model_a8, w_cfg)

    model_w8a8.save(str(dst))
    s_sz = _dir_size(src); d_sz = _dir_size(dst)
    print(f"    {src.name}: {s_sz/1e6:.1f} MB → {d_sz/1e6:.1f} MB  ({d_sz*100/s_sz:.0f}%)")
    return s_sz, d_sz


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["text", "generator", "both"],
                    default="both", nargs="?")
    ap.add_argument("-L", default="16,32,64,128,400")
    ap.add_argument("-N", default="128,256,512,1024")
    ap.add_argument("--samples", type=int, default=N_CALIB)
    args = ap.parse_args()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    total_s = total_d = 0

    if args.which in ("text", "both"):
        for L in (int(x) for x in args.L.split(",") if x.strip()):
            src = SRC_DIR / f"kitten_text_L{L}.mlpackage"
            dst = DST_DIR / f"kitten_text_L{L}.mlpackage"
            if not src.exists():
                print(f"  skip {src.name} (fp32 source missing; run convert_to_coreml first)")
                continue
            calib = _text_calibration(L, args.samples, rng)
            s, d = quantize_one(src, dst, calib)
            total_s += s; total_d += d

    if args.which in ("generator", "both"):
        for N in (int(x) for x in args.N.split(",") if x.strip()):
            src = SRC_DIR / f"kitten_generator_N{N}.mlpackage"
            dst = DST_DIR / f"kitten_generator_N{N}.mlpackage"
            if not src.exists():
                print(f"  skip {src.name} (fp32 source missing)")
                continue
            calib = _generator_calibration(N, args.samples, rng)
            s, d = quantize_one(src, dst, calib)
            total_s += s; total_d += d

    if total_s > 0:
        print(f"\ntotal: {total_s/1e6:.1f} MB → {total_d/1e6:.1f} MB  "
              f"({total_d*100/total_s:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
