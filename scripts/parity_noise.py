#!/usr/bin/env python3
"""Isolated parity test: compute_noise_contribs ggml vs torch."""
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
from torch_kitten import (  # noqa: E402
    WeightBag, AdaINResBlockHiFiGAN, compute_noise_contribs,
)

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
GGUF         = REPO_ROOT / "tmp/kitten_full.gguf"
TMP          = REPO_ROOT / "tmp"


def torch_reference(f0_proj: np.ndarray, style128: np.ndarray):
    F = f0_proj.shape[1] // 2
    w = WeightBag.load(str(SAFE_TENSORS))
    nr0_mod = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.0")
    nr1_mod = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.1")
    f0_t = torch.from_numpy(f0_proj).reshape(1, 1, 2 * F)
    s_t  = torch.from_numpy(style128).reshape(1, 128)
    with torch.no_grad():
        nr0, nr1 = compute_noise_contribs(f0_t, F, s_t, w, nr0_mod, nr1_mod, seed=None)
    return nr0.numpy().squeeze(0), nr1.numpy().squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-F", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    F = args.F
    rng = np.random.default_rng(args.seed)
    f0 = (rng.standard_normal(2 * F).astype(np.float32) * 30 + 130)
    f0[:2] = 0.0; f0[-2:] = 0.0
    f0 = f0.reshape(1, 2 * F)
    style = rng.standard_normal(128).astype(np.float32) * 0.1
    print(f"F={F}")

    print("[torch] running compute_noise_contribs …")
    ref0, ref1 = torch_reference(f0, style)
    print(f"[torch] nr0 shape={ref0.shape}, nr1 shape={ref1.shape}")

    in_path = TMP / "ns_in.bin"
    out_path = TMP / "ns_out.bin"
    with in_path.open("wb") as fp:
        fp.write(struct.pack("<i", F))
        fp.write(f0.T.astype(np.float32).tobytes())
        fp.write(style.tobytes())

    bin_path = TMP / f"kittens-tts-{args.backend}"
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", "noise",
           "--input", str(in_path), "--output", str(out_path),
           "--backend", args.backend]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout, end="")
    if r.stderr: print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0: raise SystemExit(f"exited {r.returncode}")

    with out_path.open("rb") as fp:
        hdr = fp.read(16)
        c0, l0, c1, l1 = struct.unpack("<iiii", hdr)
        raw = np.frombuffer(fp.read(), dtype=np.float32)
    n0 = c0 * l0; n1 = c1 * l1
    got0 = raw[:n0].reshape(l0, c0).T   # NLC → (C, L)
    got1 = raw[n0:].reshape(l1, c1).T

    print(f"[ggml] nr0 shape={got0.shape}, nr1 shape={got1.shape}")

    def cmp(label, ref, got):
        a, b = ref.flatten(), got.flatten()
        cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        max_err = float(np.max(np.abs(a - b)))
        mean_err = float(np.mean(np.abs(a - b)))
        print(f"[{label}] cos={cos:.6f}  max_err={max_err:.3e}  mean_err={mean_err:.3e}")
        return cos
    cos_a = cmp("nr0", ref0, got0)
    cos_b = cmp("nr1", ref1, got1)
    if cos_a >= 0.99 and cos_b >= 0.99: print("PASS")
    else: print("FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
