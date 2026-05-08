#!/usr/bin/env python3
"""Parity test: kittens-tts (ggml) generator-front vs torch reference.

The generator-front consumes already-length-regulated prosody (256, F) and
produces (f0_proj, n_proj) each shape (1, 2F). We synthesize a fixed input
and compare against torch_kitten's GeneratorStage front-half (the slice
that ends after f0_proj/n_proj).
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
from torch_kitten import (  # noqa: E402
    WeightBag, AdaINResBlock1D, load_onnx_bidir_lstm, load_conv1d, conv1d_ncl,
)

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
GGUF         = REPO_ROOT / "tmp/kitten_full.gguf"
TMP          = REPO_ROOT / "tmp"


def torch_reference(prosody_lr: np.ndarray, style128: np.ndarray):
    """Replicate the front half of torch_kitten.GeneratorStage.

    Inputs:
        prosody_lr: (256, F) numpy NCL representation
        style128:   (128,) prosodic style
    Returns:
        (f0_proj_np, n_proj_np)  each shape (1, 2F)
    """
    w = WeightBag.load(str(SAFE_TENSORS))

    F = prosody_lr.shape[1]
    pr_t = torch.from_numpy(prosody_lr).reshape(1, 256, F)
    style_t = torch.from_numpy(style128).reshape(1, 128)

    shared_lstm = load_onnx_bidir_lstm(
        w, "onnx::LSTM_6020", "onnx::LSTM_6021", "onnx::LSTM_6019")

    shared_in = pr_t.permute(2, 0, 1).contiguous()  # (F, 1, 256)
    sy = shared_lstm(shared_in)                     # (F, 2, 1, H)
    fn_lstm_nlc = sy.permute(2, 0, 1, 3).reshape(1, F, 128)
    fn_in_ncl = fn_lstm_nlc.transpose(1, 2).contiguous()   # (1, 128, F)

    f0_0 = AdaINResBlock1D(w, "predictor.F0.0", upsample=False, divide=True)
    f0_1 = AdaINResBlock1D(w, "predictor.F0.1", upsample=True,  divide=True)
    f0_2 = AdaINResBlock1D(w, "predictor.F0.2", upsample=False, divide=True)
    n_0  = AdaINResBlock1D(w, "predictor.N.0",  upsample=False, divide=True)
    n_1  = AdaINResBlock1D(w, "predictor.N.1",  upsample=True,  divide=True)
    n_2  = AdaINResBlock1D(w, "predictor.N.2",  upsample=False, divide=True)
    f0pW, f0pB = load_conv1d(w, "predictor.F0_proj")
    npW, npB   = load_conv1d(w, "predictor.N_proj")

    with torch.no_grad():
        f0 = f0_0(fn_in_ncl, style_t, shortcut_input=fn_in_ncl)
        f0 = f0_1(f0,        style_t, shortcut_input=f0)
        f0 = f0_2(f0,        style_t, shortcut_input=f0)
        f0p = conv1d_ncl(f0, f0pW, f0pB if f0pB is not None else torch.zeros(f0pW.shape[0]),
                         padding=(f0pW.shape[-1] - 1) // 2)

        nx = n_0(fn_in_ncl, style_t, shortcut_input=fn_in_ncl)
        nx = n_1(nx,        style_t, shortcut_input=nx)
        nx = n_2(nx,        style_t, shortcut_input=nx)
        np_ = conv1d_ncl(nx, npW, npB if npB is not None else torch.zeros(npW.shape[0]),
                         padding=(npW.shape[-1] - 1) // 2)
    return f0p.numpy().reshape(2 * F), np_.numpy().reshape(2 * F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-F", type=int, default=24, help="number of frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    F = args.F
    rng = np.random.default_rng(args.seed)
    prosody_lr = rng.standard_normal((256, F)).astype(np.float32) * 0.1
    style = rng.standard_normal(128).astype(np.float32) * 0.1
    print(f"F={F}")

    print("[torch] running reference …")
    ref_f0, ref_n = torch_reference(prosody_lr, style)
    print(f"[torch] f0[:6]={ref_f0[:6]}  n[:6]={ref_n[:6]}")

    in_path = TMP / "gf_in.bin"
    out_path = TMP / "gf_out.bin"
    with in_path.open("wb") as f:
        f.write(struct.pack("<i", F))
        # NCL ne=(256, F) numpy already matches ggml NLC ne=(256, F) — both
        # have data[c*F + l] in linear order.  Wait: numpy (256, F) C-order
        # is data[c*F + l]; ggml NLC ne=(256, F) is data[l*256 + c]. DIFFERENT!
        # Save numpy-row-major NCL as data[c*F + l] = ref's layout. The C
        # side reads it into a tensor of shape (256, F) — interpreting the
        # bytes via ggml ne[0]=256 fastest gives ggml NLC data[l*256+c].
        # That doesn't match the saved layout. We need to .T before save so
        # that numpy linear is data[l*256+c].
        f.write(prosody_lr.T.astype(np.float32).tobytes())   # save (F, 256) C-order
        f.write(style.tobytes())

    bin_path = TMP / f"kittens-tts-{args.backend}"
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", "genfront",
           "--input", str(in_path), "--output", str(out_path),
           "--backend", args.backend]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout, end="")
    if r.stderr: print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0: raise SystemExit(f"exited {r.returncode}")

    raw = np.fromfile(out_path, dtype=np.float32)
    if raw.size != 4 * F: raise SystemExit(f"size mismatch {raw.size} vs {4*F}")
    got_f0 = raw[:2*F]
    got_n  = raw[2*F:]

    def cmp(label, ref, got):
        a, b = ref.flatten(), got.flatten()
        cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        max_err = float(np.max(np.abs(a - b)))
        mean_err = float(np.mean(np.abs(a - b)))
        print(f"[{label}] cos={cos:.6f}  max_err={max_err:.3e}  mean_err={mean_err:.3e}")
        return cos
    cos_f = cmp("f0_proj", ref_f0, got_f0)
    cos_n = cmp("n_proj",  ref_n,  got_n)
    if cos_f >= 0.999 and cos_n >= 0.999:
        print("PASS")
    else:
        print("FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
