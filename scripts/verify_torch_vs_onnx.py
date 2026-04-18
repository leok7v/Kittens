#!/usr/bin/env python3
"""Run the same (input_ids, style, speed) through ONNX Runtime and the
PyTorch port in scripts/torch_kitten.py and compare outputs.

The ONNX model and the PyTorch port both contain non-deterministic noise
(RandomUniformLike + RandomNormalLike) in the SineGen path, so the waveforms
will NOT match exactly even when the port is correct. We compare:
  - duration: must match exactly (deterministic path).
  - waveform: expect correlated but not identical; print RMS error, peak
    diff, and length match.

Running end-to-end first lets us decide if we need to expose intermediate
tensors for bisection.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import KittenTTS, WeightBag  # noqa: E402


def load_voice(voices_path: Path, voice_id: str) -> np.ndarray:
    v = np.load(str(voices_path))
    assert voice_id in v.files, f"voice {voice_id!r} not in {list(v.files)}"
    return v[voice_id]


def synth_inputs(voices_path: Path, voice_id: str, n_phonemes: int, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize fake but valid inputs for smoke testing.

    Kitten's phoneme vocabulary has 178 entries (embedding shape [178,128]).
    We pick something well within range so onnx doesn't throw.
    """
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(1, 170, size=(1, n_phonemes), dtype=np.int64)
    voices = load_voice(voices_path, voice_id)
    # Use row n_phonemes (clamped) as the style vector — matches upstream logic.
    row = min(n_phonemes, voices.shape[0] - 1)
    style = voices[row : row + 1].astype(np.float32)  # (1, 256)
    speed = np.array([1.0], dtype=np.float32)
    return input_ids, style, speed


def run_onnx(onnx_path: Path, input_ids, style, speed) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    t0 = time.time()
    outs = sess.run(None, {"input_ids": input_ids, "style": style, "speed": speed})
    dt = time.time() - t0
    out_names = [o.name for o in sess.get_outputs()]
    return {name: out for name, out in zip(out_names, outs)} | {"_elapsed_s": dt}


def run_torch(weights_path: Path, input_ids, style, speed, seed: int = 0
              ) -> dict[str, np.ndarray]:
    w = WeightBag.load(weights_path)
    model = KittenTTS(w).eval()
    with torch.no_grad():
        ids_t = torch.from_numpy(input_ids).long()
        style_t = torch.from_numpy(style).float()
        torch.manual_seed(seed)
        t0 = time.time()
        wav, durs = model(ids_t, style_t, speed=float(speed[0]), noise_seed=seed)
        dt = time.time() - t0
    return {
        "waveform": wav.cpu().numpy(),
        "duration": durs.cpu().numpy().astype(np.int64),
        "_elapsed_s": dt,
    }


def compare(onnx_out: dict, torch_out: dict) -> int:
    rc = 0

    d_onnx = onnx_out["duration"]
    d_torch = torch_out["duration"]
    print(f"ONNX duration  shape={d_onnx.shape}  sum={int(d_onnx.sum())}")
    print(f"Torch duration shape={d_torch.shape}  sum={int(d_torch.sum())}")
    if d_onnx.shape != d_torch.shape:
        print(f"  ! duration shape mismatch")
        rc = 1
    else:
        diff = (d_onnx - d_torch)
        print(f"  duration max-abs diff = {int(np.abs(diff).max())}")
        if not np.array_equal(d_onnx, d_torch):
            print(f"  onnx  = {d_onnx.tolist()}")
            print(f"  torch = {d_torch.tolist()}")

    w_onnx = onnx_out["waveform"].astype(np.float64)
    w_torch = torch_out["waveform"].astype(np.float64)
    print(f"\nONNX waveform  shape={w_onnx.shape}  rms={np.sqrt((w_onnx**2).mean()):.4g}")
    print(f"Torch waveform shape={w_torch.shape}  rms={np.sqrt((w_torch**2).mean()):.4g}")
    n = min(w_onnx.size, w_torch.size)
    if n > 0:
        diff = w_onnx[:n] - w_torch[:n]
        rms = np.sqrt((diff ** 2).mean())
        peak = np.abs(diff).max()
        num = (w_onnx[:n] * w_torch[:n]).sum()
        den = np.sqrt((w_onnx[:n] ** 2).sum() * (w_torch[:n] ** 2).sum()) + 1e-30
        cos = num / den
        print(f"  waveform rms diff = {rms:.4g}")
        print(f"  waveform peak diff = {peak:.4g}")
        print(f"  waveform cos sim  = {cos:.6f}   (1.0 = identical)")
    print(f"\nonnx elapsed  = {onnx_out['_elapsed_s']:.3f}s")
    print(f"torch elapsed = {torch_out['_elapsed_s']:.3f}s")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path,
                    default=Path("scripts/models/kitten_tts_nano_v0_8.onnx"))
    ap.add_argument("--safetensors", type=Path,
                    default=Path("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"))
    ap.add_argument("--voices", type=Path,
                    default=Path("scripts/models/voices.npz"))
    ap.add_argument("--voice", default="expr-voice-5-m")
    ap.add_argument("--n-phonemes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_ids, style, speed = synth_inputs(args.voices, args.voice, args.n_phonemes, args.seed)
    print(f"input_ids shape={input_ids.shape}  values={input_ids[0, :10].tolist()}...")

    print("\n-- ONNX Runtime --")
    onnx_out = run_onnx(args.onnx, input_ids, style, speed)

    print("\n-- PyTorch port --")
    torch_out = run_torch(args.safetensors, input_ids, style, speed, seed=args.seed)

    print("\n== compare ==")
    return compare(onnx_out, torch_out)


if __name__ == "__main__":
    sys.exit(main())
