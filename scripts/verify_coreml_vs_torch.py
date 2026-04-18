#!/usr/bin/env python3
"""Compare CoreML (.mlpackage) output against the PyTorch reference.

Runs TextStage in CoreML, builds the alignment matrix + length-regulates in
numpy, runs GeneratorStage in CoreML, then does the same pipeline in PyTorch
for comparison. ONNX parity is already verified in verify_torch_vs_onnx.py,
so this script isolates conversion loss.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import TextStage, GeneratorStage, WeightBag  # noqa: E402


def build_alignment(durs: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    n_frames = int(durs.sum())
    align = np.zeros((1, L, n_frames), dtype=np.float32)
    j = 0
    for i, d in enumerate(durs.tolist()):
        align[0, i, j:j+int(d)] = 1.0
        j += int(d)
    return align, n_frames


def durs_from_dur_sig(dur_sig: np.ndarray, speed: float = 1.0) -> np.ndarray:
    dur_sum = dur_sig.sum(axis=-1)[0]
    dur_scaled = dur_sum / speed
    return np.maximum(1, np.round(dur_scaled).astype(np.int32))


def pad_to(x: np.ndarray, axis: int, n: int) -> np.ndarray:
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, max(0, n - x.shape[axis]))
    y = np.pad(x, pad)
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(0, n)
    return y[tuple(slicer)]


def compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.shape != b.shape:
        print(f"[{label}] shape mismatch coreml={a.shape} torch={b.shape}")
    a = a.flatten(); b = b.flatten()
    n = min(a.size, b.size)
    diff = a[:n] - b[:n]
    rms = float(np.sqrt((diff**2).mean()))
    peak = float(np.max(np.abs(diff)))
    num = float((a[:n]*b[:n]).sum())
    den = float(np.sqrt((a[:n]**2).sum()*(b[:n]**2).sum()) + 1e-30)
    cos = num / den
    print(f"[{label}] rms={rms:.4g} peak={peak:.4g} cos={cos:.6f} "
          f"coreml_rms={np.sqrt((a**2).mean()):.4g} torch_rms={np.sqrt((b**2).mean()):.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-L", type=int, default=128)
    ap.add_argument("--voice", default="expr-voice-5-m")
    ap.add_argument("--safetensors", default="Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    ap.add_argument("--voices", default="scripts/models/voices.npz")
    args = ap.parse_args()

    import coremltools as ct
    text_pkg = f"scripts/models/kitten_text_L{args.L}.mlpackage"
    print(f"loading {text_pkg}")
    text_ml = ct.models.MLModel(text_pkg)

    rng = np.random.default_rng(0)
    input_ids_np = rng.integers(1, 170, size=(1, args.L), dtype=np.int32)
    input_ids_t = torch.from_numpy(input_ids_np.astype(np.int64))
    voices = np.load(args.voices)
    style_np = voices[args.voice][args.L:args.L+1].astype(np.float32)
    style_t = torch.from_numpy(style_np)

    # PyTorch reference.
    w = WeightBag.load(args.safetensors)
    text_stage = TextStage(w).eval()
    gen_stage = GeneratorStage(w).eval()

    # Full attention mask (no padding) for parity testing.
    mask_np = np.ones((1, args.L), dtype=np.float32)
    mask_t = torch.from_numpy(mask_np)

    with torch.no_grad():
        t0 = time.time()
        p_t, tf_t, ds_t = text_stage(input_ids_t, style_t, mask_t)
        ttext = time.time() - t0

    # CoreML TextStage.
    t0 = time.time()
    text_out = text_ml.predict({
        "input_ids": input_ids_np,
        "style": style_np,
        "attention_mask": mask_np,
    })
    ttext_ml = time.time() - t0

    # Find outputs by shape heuristic (coremltools renames outputs).
    by_shape: dict[tuple[int, ...], np.ndarray] = {}
    for k, v in text_out.items():
        by_shape[v.shape] = v
    p_ml = by_shape.get((1, 256, args.L))
    tf_ml = by_shape.get((1, 128, args.L))
    ds_ml = by_shape.get((1, args.L, 50))
    assert p_ml is not None and tf_ml is not None and ds_ml is not None, \
        f"expected 3 tensors with known shapes, got {[v.shape for v in text_out.values()]}"

    compare("TextStage prosody_ncl", p_ml, p_t.numpy())
    compare("TextStage text_features", tf_ml, tf_t.numpy())
    compare("TextStage dur_sig", ds_ml, ds_t.numpy())

    # Build alignment from torch's dur_sig (same either way; the CoreML one
    # might yield a different nFrames due to fp32 roundoff).
    durs_t = durs_from_dur_sig(ds_t.numpy())
    durs_ml = durs_from_dur_sig(ds_ml)
    print(f"\ndurs torch={durs_t.tolist()}")
    print(f"durs coreml={durs_ml.tolist()}")

    align, n_frames = build_alignment(durs_t, args.L)
    print(f"nFrames={n_frames}")

    # Expand. CoreML generator is fixed at N=256; pad/trim nFrames to 256 for testing.
    gen_pkg = "scripts/models/kitten_generator_N256.mlpackage"
    print(f"\nloading {gen_pkg}")
    gen_ml = ct.models.MLModel(gen_pkg)
    target_nf = 256

    prosody_lr_np = p_ml @ align           # (1, 256, nFrames)
    text_lr_np = tf_ml @ align             # (1, 128, nFrames)
    # Pad or truncate to the model's expected nFrames.
    prosody_lr_np = pad_to(prosody_lr_np, 2, target_nf)
    text_lr_np = pad_to(text_lr_np, 2, target_nf)

    prosody_lr_t = torch.from_numpy(prosody_lr_np).float()
    text_lr_t = torch.from_numpy(text_lr_np).float()

    torch.manual_seed(0)
    with torch.no_grad():
        wav_t = gen_stage(prosody_lr_t, text_lr_t, style_t).numpy()

    gen_out = gen_ml.predict({
        "prosody_lr": prosody_lr_np,
        "text_lr": text_lr_np,
        "style": style_np,
    })
    wav_ml = next(iter(gen_out.values()))
    print(f"torch wav shape={wav_t.shape}  coreml wav shape={wav_ml.shape}")
    compare("GeneratorStage waveform", wav_ml, wav_t)


if __name__ == "__main__":
    main()
