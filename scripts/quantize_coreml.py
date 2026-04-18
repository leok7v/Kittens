#!/usr/bin/env python3
"""Quantize CoreML .mlpackages to INT8 weights (activations stay fp16 on ANE).

Reuses our existing fp32 .mlpackage files produced by convert_to_coreml.py and
emits compressed versions. The model math is unchanged — each weight tensor is
stored as int8 + (scale, zero_point) per channel and dequantized at load time.

Weight compression ≈ 4× smaller per bucket (fp32 → int8). Bundle size for the
7 shipped buckets should drop from ~195 MB to ~55 MB.

Typical quality impact for weight-only quant is modest (< 3 dB MCD) for conv
nets; LSTM-heavy models can be more sensitive, which is why the text buckets
are worth listen-testing before replacing.

Usage:
    python scripts/quantize_coreml.py
    python scripts/quantize_coreml.py --input-dir Sources/KittenApp/Resources/coreml \\
                                      --output-dir scripts/models/quantized
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)


def quantize_one(src: Path, dst: Path,
                 mode: str = "linear_symmetric",
                 granularity: str = "per_channel") -> tuple[int, int]:
    """Quantize one .mlpackage from src to dst. Returns (src_bytes, dst_bytes)."""
    if dst.exists():
        shutil.rmtree(dst)
    print(f"  loading {src.name}")
    model = ct.models.MLModel(str(src))
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode=mode,
            dtype="int8",
            granularity=granularity,
        )
    )
    print(f"    quantizing (mode={mode}, granularity={granularity}) ...")
    q = linear_quantize_weights(model, config=config)
    q.save(str(dst))
    src_sz = _dir_size(src)
    dst_sz = _dir_size(dst)
    print(f"    {src.name}: {src_sz/1e6:.1f} MB  →  {dst_sz/1e6:.1f} MB  "
          f"({dst_sz*100/src_sz:.0f}%)")
    return src_sz, dst_sz


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path,
                    default=Path("Sources/KittenApp/Resources/coreml"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("scripts/models/quantized"))
    ap.add_argument("--mode", default="linear_symmetric",
                    choices=["linear", "linear_symmetric"])
    ap.add_argument("--granularity", default="per_channel",
                    choices=["per_tensor", "per_channel"])
    ap.add_argument("--only", default=None,
                    help="only quantize packages whose name contains this string")
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"error: {args.input_dir} not found", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packages = sorted(args.input_dir.glob("*.mlpackage"))
    if args.only:
        packages = [p for p in packages if args.only in p.name]
    if not packages:
        print(f"error: no .mlpackage in {args.input_dir}", file=sys.stderr)
        return 1

    total_src = total_dst = 0
    for src in packages:
        dst = args.output_dir / src.name
        s, d = quantize_one(src, dst, mode=args.mode, granularity=args.granularity)
        total_src += s
        total_dst += d
    print(f"\ntotal: {total_src/1e6:.1f} MB  →  {total_dst/1e6:.1f} MB  "
          f"({total_dst*100/total_src:.0f}%)  across {len(packages)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
