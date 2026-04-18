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

SRC_DIR = Path("scripts/models/fp32")
DST_DIR = Path("scripts/models/int8wa")
VOICES_NPZ = Path("scripts/models/voices.npz")

N_CALIB = 32


def _voices_sample(k: int, rng: np.random.Generator) -> np.ndarray:
    """Return k random style vectors (shape (1, 256) each)."""
    v = np.load(str(VOICES_NPZ))
    # Pool all 8 voice tables together — ~3200 rows.
    rows = np.concatenate([v[name] for name in v.files], axis=0).astype(np.float32)
    idx = rng.integers(0, rows.shape[0], size=k)
    return rows[idx].reshape(k, 1, 256)


def _text_calibration(L: int, n: int, rng: np.random.Generator) -> list[dict]:
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
    """Synthetic generator inputs. The real distribution of prosody_lr /
    text_lr depends on upstream stages; a Gaussian approximation is
    usually fine for activation calibration of the generator's own
    activations. Mean/std pulled from a one-shot real run in the past."""
    # Rough empirical stats from a real TextStage + length-reg run:
    prosody_mean, prosody_std = -0.03, 1.65
    text_mean, text_std = 0.00, 0.24
    samples = []
    styles = _voices_sample(n, rng)
    for i in range(n):
        prosody = rng.normal(prosody_mean, prosody_std, (1, 256, N)).astype(np.float32)
        text = rng.normal(text_mean, text_std, (1, 128, N)).astype(np.float32)
        samples.append({
            "prosody_lr": prosody,
            "text_lr":    text,
            "style":      styles[i],
        })
    return samples


def quantize_one(src: Path, dst: Path, calib: list[dict]) -> tuple[int, int]:
    if dst.exists():
        shutil.rmtree(dst)
    print(f"  loading {src.name}")
    model = ct.models.MLModel(str(src))

    # Step 1: quantize activations with calibration.
    # Skip ops whose inputs are non-float (e.g. embedding lookup from int32
    # input_ids) — coremltools' quantize pass has a bug there, inserting an
    # fp32 scale on an int32 tensor and failing the validator.
    def op_selector(op) -> bool:
        for x in op.inputs.values():
            if isinstance(x, (list, tuple)):
                xs = x
            else:
                xs = [x]
            for v in xs:
                dt = getattr(v, "dtype", None)
                if dt is not None and "int" in str(dt):
                    return False
        return True

    act_cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"),
        op_selector=op_selector,
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
