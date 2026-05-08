#!/usr/bin/env python3
"""Parity test: kittens-tts (ggml) generator+iSTFT vs torch reference.

Uses zero noise tensors (skipping compute_noise_contribs for now).
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from torch_kitten import WeightBag, GeneratorPipeline  # noqa: E402

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
GGUF         = REPO_ROOT / "tmp/kitten_full.gguf"
TMP          = REPO_ROOT / "tmp"


def torch_reference(dec_out: np.ndarray, nr0: np.ndarray, nr1: np.ndarray,
                    style128: np.ndarray) -> np.ndarray:
    F = dec_out.shape[1] // 2
    w = WeightBag.load(str(SAFE_TENSORS))
    gen = GeneratorPipeline(w).eval()

    dec_t = torch.from_numpy(dec_out).reshape(1, 256, 2 * F)
    nr0_t = torch.from_numpy(nr0).reshape(1, 128, 20 * F)
    nr1_t = torch.from_numpy(nr1).reshape(1, 64,  120 * F + 1)
    s_t   = torch.from_numpy(style128).reshape(1, 128)
    with torch.no_grad():
        audio = gen(dec_t, s_t, nr0_t, nr1_t)
    return audio.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-F", type=int, default=8)   # small F to keep test fast
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    F = args.F
    rng = np.random.default_rng(args.seed)
    dec_out = rng.standard_normal((256, 2 * F)).astype(np.float32) * 0.5
    nr0 = np.zeros((128, 20 * F), dtype=np.float32)
    nr1 = np.zeros((64, 120 * F + 1), dtype=np.float32)
    style = rng.standard_normal(128).astype(np.float32) * 0.1
    print(f"F={F}, expected audio length = {600*F}")

    print("[torch] running GeneratorPipeline (zero noise)…")
    ref_audio = torch_reference(dec_out, nr0, nr1, style)
    print(f"[torch] audio shape={ref_audio.shape}  audio[:6]={ref_audio[:6]}")

    in_path = TMP / "gen_in.bin"
    out_path = TMP / "gen_out.bin"
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", F))
        # NLC layout for ggml: numpy (C, L) → save .T (gives data[l*C+c])
        f.write(dec_out.T.astype(np.float32).tobytes())
        f.write(nr0.T.astype(np.float32).tobytes())
        f.write(nr1.T.astype(np.float32).tobytes())
        f.write(style.tobytes())

    bin_path = TMP / f"kittens-tts-{args.backend}"
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", "generator",
           "--input", str(in_path), "--output", str(out_path),
           "--backend", args.backend]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout, end="")
    if r.stderr: print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0: raise SystemExit(f"exited {r.returncode}")

    with out_path.open("rb") as f:
        T_g, = struct.unpack("<i", f.read(4))
        got = np.frombuffer(f.read(), dtype=np.float32)
    print(f"[ggml]  audio T={T_g}  audio[:6]={got[:6]}")

    if T_g != ref_audio.size:
        print(f"LENGTH MISMATCH: got {T_g} expected {ref_audio.size}")
        sys.exit(1)

    a, b = ref_audio.flatten(), got.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    max_err = float(np.max(np.abs(a - b)))
    mean_err = float(np.mean(np.abs(a - b)))
    print(f"[generator] cos={cos:.6f}  max_err={max_err:.3e}  mean_err={mean_err:.3e}")
    if cos < 0.99:
        print("FAIL"); sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
