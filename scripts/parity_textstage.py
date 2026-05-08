#!/usr/bin/env python3
"""Parity test: kittens-tts (ggml) TextStage vs torch_kitten.TextStage.

Compares prosody_ncl, text_features_ncl, dur_sig outputs.
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
from torch_kitten import TextStage, WeightBag  # noqa: E402

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
GGUF         = REPO_ROOT / "tmp/kitten_full.gguf"
TMP          = REPO_ROOT / "tmp"


def load_voices_style() -> np.ndarray:
    """Pick a fixed style row (256-d) from voices.npz; ref id 50."""
    v = np.load(REPO_ROOT / "scripts/models/voices.npz")
    arr = v["expr-voice-5-m"]
    return arr[50].astype(np.float32)   # (256,)


def torch_reference(ids: np.ndarray, style: np.ndarray):
    w = WeightBag.load(str(SAFE_TENSORS))
    ts = TextStage(w).eval()
    L = ids.shape[0]
    ids_t = torch.from_numpy(ids.astype(np.int64)).reshape(1, L)
    style_t = torch.from_numpy(style).reshape(1, 256)
    mask = torch.ones(1, L, dtype=torch.float32)
    with torch.no_grad():
        prosody, text, dur = ts(ids_t, style_t, mask)
    # prosody: (1, 256, L), text: (1, 128, L), dur: (1, L, 50)
    return (prosody.numpy().reshape(256, L),
            text.numpy().reshape(128, L),
            dur.numpy().reshape(L, 50))


def write_input(path: Path, ids: np.ndarray, style: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<i", int(ids.shape[0])))
        f.write(ids.astype(np.int32).tobytes())
        f.write(style.astype(np.float32).tobytes())


def read_outputs(path: Path, L: int):
    raw = np.fromfile(path, dtype=np.float32)
    p_n = 256 * L
    t_n = 128 * L
    d_n = 50 * L
    if raw.size != p_n + t_n + d_n:
        raise SystemExit(f"output size mismatch: {raw.size} vs {p_n+t_n+d_n}")
    # All three outputs are NLC (ne[0]=channels, ne[1]=L). For ggml NLC,
    # linear data is data[l*C + c]. numpy.reshape(L, C) gives that order:
    #   numpy[l, c] = raw[l*C + c]
    # Then .T gives shape (C, L) for direct compare with torch (C, L) flat.
    p = raw[:p_n].reshape(L, 256).T               # (256, L)
    t = raw[p_n:p_n+t_n].reshape(L, 128).T        # (128, L)
    d = raw[p_n+t_n:].reshape(L, 50)              # (L, 50)
    return p, t, d


def cmp(label: str, ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    a, b = ref.flatten(), got.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    max_err = float(np.max(np.abs(a - b)))
    mean_err = float(np.mean(np.abs(a - b)))
    print(f"[{label}] cos={cos:.6f}  max_err={max_err:.3e}  mean_err={mean_err:.3e}")
    return cos, max_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-L", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    L = args.L
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(1, 178, size=L, dtype=np.int32)
    style = load_voices_style()
    print(f"L={L}  ids[:6]={ids[:6].tolist()}")

    # torch reference
    print("[torch] running TextStage…")
    ref_p, ref_t, ref_d = torch_reference(ids, style)
    print(f"[torch] prosody[:4,0]={ref_p[:4,0]}  text[:4,0]={ref_t[:4,0]}  dur[0,:6]={ref_d[0,:6]}")

    in_path = TMP / "ts_in.bin"
    out_path = TMP / "ts_out.bin"
    write_input(in_path, ids, style)

    bin_path = TMP / f"kittens-tts-{args.backend}"
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", "textstage",
           "--input", str(in_path), "--output", str(out_path),
           "--backend", args.backend]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout, end="")
    if r.stderr: print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0:
        raise SystemExit(f"exited {r.returncode}")

    got_p, got_t, got_d = read_outputs(out_path, L)
    print(f"[ggml]  prosody[:4,0]={got_p[:4,0]}  text[:4,0]={got_t[:4,0]}  dur[0,:6]={got_d[0,:6]}")

    cos_p, _ = cmp("prosody", ref_p, got_p)
    cos_t, _ = cmp("text",    ref_t, got_t)
    cos_d, _ = cmp("dur_sig", ref_d, got_d)

    PASS = 0.999
    fails = [n for n,c in [("prosody",cos_p),("text",cos_t),("dur_sig",cos_d)] if c < PASS]
    if fails:
        print(f"FAIL: {fails}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
