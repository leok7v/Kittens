#!/usr/bin/env python3
"""List every tensor in the KittenTTS safetensors + voices.npz.

Used to cross-check weight names against the MLX port and confirm LSTM
layouts / dequantization patterns before porting to PyTorch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open


def fmt_shape(shape):
    return "[" + ",".join(str(d) for d in shape) + "]"


def dump_safetensors(path: Path, grep: str | None) -> None:
    with safe_open(str(path), framework="pt") as f:
        keys = sorted(f.keys())
        if grep:
            keys = [k for k in keys if grep in k]
        for k in keys:
            t = f.get_tensor(k)
            print(f"  {k:80s} {str(t.dtype):16s} {fmt_shape(t.shape)}")
        print(f"total: {len(keys)} tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default="Sources/KittenTTS/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    ap.add_argument("--voices", default="scripts/models/voices.npz")
    ap.add_argument("--grep", default=None, help="substring filter")
    args = ap.parse_args()

    print(f"=== {args.safetensors} ===")
    dump_safetensors(Path(args.safetensors), args.grep)

    print(f"\n=== {args.voices} ===")
    v = np.load(args.voices)
    for k in sorted(v.files):
        a = v[k]
        print(f"  {k:40s} {str(a.dtype):10s} {fmt_shape(a.shape)}")


if __name__ == "__main__":
    main()
