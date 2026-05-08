#!/usr/bin/env python3
"""Parity test: kittens-tts (ggml) decoder pipeline vs torch reference."""
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
from torch_kitten import WeightBag, DecoderPipeline  # noqa: E402

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
GGUF         = REPO_ROOT / "tmp/kitten_full.gguf"
TMP          = REPO_ROOT / "tmp"


def torch_reference(text_lr: np.ndarray, f0_proj: np.ndarray, n_proj: np.ndarray,
                    style128: np.ndarray) -> np.ndarray:
    """text_lr: (128, F),  f0/n_proj: (1, 2F),  style128: (128,)
    Returns: numpy (C_out, L_out)
    """
    F = text_lr.shape[1]
    w = WeightBag.load(str(SAFE_TENSORS))
    dec = DecoderPipeline(w).eval()

    text_lr_t = torch.from_numpy(text_lr).reshape(1, 128, F)
    f0_t = torch.from_numpy(f0_proj).reshape(1, 1, 2*F)
    n_t  = torch.from_numpy(n_proj).reshape(1, 1, 2*F)
    s_t  = torch.from_numpy(style128).reshape(1, 128)
    with torch.no_grad():
        out = dec(text_lr_t, f0_t, n_t, s_t)   # (1, C_out, L_out)
    return out.numpy().reshape(out.shape[1], out.shape[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-F", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    F = args.F
    rng = np.random.default_rng(args.seed)
    text_lr = rng.standard_normal((128, F)).astype(np.float32) * 0.1
    f0_proj = rng.standard_normal((1, 2*F)).astype(np.float32) * 50.0
    n_proj  = rng.standard_normal((1, 2*F)).astype(np.float32) * 5.0
    style   = rng.standard_normal(128).astype(np.float32) * 0.1
    print(f"F={F}")

    print("[torch] running DecoderPipeline …")
    ref = torch_reference(text_lr, f0_proj, n_proj, style)
    C_out, L_out = ref.shape
    print(f"[torch] ref shape=({C_out}, {L_out})  ref[:4,0]={ref[:4,0]}")

    in_path = TMP / "dec_in.bin"
    out_path = TMP / "dec_out.bin"
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", F))
        # ggml NLC ne=(C, L) needs data[l*C+c] in linear. numpy (C, L) row-major
        # is data[c*L+l]. We need to .T so numpy linear matches ggml.
        f.write(text_lr.T.astype(np.float32).tobytes())
        f.write(f0_proj.T.astype(np.float32).tobytes())
        f.write(n_proj.T.astype(np.float32).tobytes())
        f.write(style.astype(np.float32).tobytes())

    bin_path = TMP / f"kittens-tts-{args.backend}"
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", "decoder",
           "--input", str(in_path), "--output", str(out_path),
           "--backend", args.backend]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout, end="")
    if r.stderr: print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0: raise SystemExit(f"exited {r.returncode}")

    # Read header + data
    with out_path.open("rb") as f:
        hdr = f.read(8)
        c_g, l_g = struct.unpack("<ii", hdr)
        raw = np.frombuffer(f.read(), dtype=np.float32)
    print(f"[ggml]  out shape=({c_g}, {l_g})  total floats={raw.size}")

    if (c_g, l_g) != (C_out, L_out):
        print(f"SHAPE MISMATCH: got ({c_g},{l_g}) expected ({C_out},{L_out})")
        sys.exit(1)

    # NLC ne=(C, L) linear = data[l*C+c]. To get torch-shape (C, L):
    # numpy.reshape(L, C).T → numpy[c, l] = raw[l*C+c]. ✓
    got = raw.reshape(l_g, c_g).T

    a, b = ref.flatten(), got.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    max_err = float(np.max(np.abs(a - b)))
    mean_err = float(np.mean(np.abs(a - b)))
    print(f"[decoder] cos={cos:.6f}  max_err={max_err:.3e}  mean_err={mean_err:.3e}")
    if cos < 0.999:
        print("FAIL"); sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
