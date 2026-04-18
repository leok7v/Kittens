#!/usr/bin/env python3
"""Plan B: merge per-bucket .mlpackages into multifunction .mlpackages.

Weights are deduplicated at save time, so the resulting file is roughly
the size of a single bucket — while still accepting every traced shape.
Swift picks the function by name at load time based on input length.

Final layout:
    Sources/KittenApp/Resources/coreml/
        kitten_text_int8w.mlpackage           functions: L_16, L_32, L_64, L_128, L_400
        kitten_text_fp16.mlpackage            same shapes, fp16 weights
        kitten_text_fp32.mlpackage            same shapes, fp32 weights
        kitten_text_int8wa.mlpackage          same shapes, int8 weights + int8 activations
        kitten_generator_int8w.mlpackage      functions: N_128, N_256, N_512, N_1024
        kitten_generator_fp16.mlpackage       same shapes, fp16
        kitten_generator_fp32.mlpackage       same shapes, fp32
        kitten_generator_int8wa.mlpackage     same shapes, int8wa

Input buckets are expected under scripts/models/ with per-variant subdirs:
    scripts/models/fp32/kitten_text_L16.mlpackage, ...
    scripts/models/fp16/kitten_text_L16.mlpackage, ...
    scripts/models/int8w/kitten_text_L16.mlpackage, ...  (alias = current scripts/models/quantized/)
    scripts/models/int8wa/kitten_text_L16.mlpackage, ... (from calibration step)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

TEXT_BUCKETS = [16, 32, 64, 128, 400]
GEN_BUCKETS = [128, 256, 512, 1024]


def merge_stage(stage: str, buckets: list[int], per_variant_dir: Path,
                out_path: Path, default_bucket: int) -> None:
    """Merge one stage (text or generator) × one variant into multifunction."""
    desc = MultiFunctionDescriptor()
    axis = "L" if stage == "text" else "N"
    for size in buckets:
        pkg = per_variant_dir / f"kitten_{stage}_{axis}{size}.mlpackage"
        if not pkg.exists():
            print(f"  ⚠ skipping missing {pkg.name}")
            continue
        desc.add_function(
            model_path=str(pkg),
            src_function_name="main",
            target_function_name=f"{axis}_{size}",
        )
        print(f"  + {pkg.name} as {axis}_{size}")
    desc.default_function_name = f"{axis}_{default_bucket}"
    if out_path.exists():
        shutil.rmtree(out_path)
    save_multifunction(desc, str(out_path))
    print(f"  saved {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["fp32", "fp16", "int8w", "int8wa"])
    ap.add_argument("--input-dir", type=Path, default=None,
                    help="per-variant directory of bucket mlpackages "
                         "(default: scripts/models/<variant>/)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("scripts/models/multifunction"))
    args = ap.parse_args()

    src = args.input_dir or Path("scripts/models") / args.variant
    if not src.exists():
        print(f"error: {src} does not exist. "
              f"Run convert_to_coreml.py (+ quantize_coreml.py) first.",
              file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== text  variant={args.variant} ===")
    merge_stage("text", TEXT_BUCKETS, src,
                args.out_dir / f"kitten_text_{args.variant}.mlpackage",
                default_bucket=128)

    print(f"\n=== generator  variant={args.variant} ===")
    merge_stage("generator", GEN_BUCKETS, src,
                args.out_dir / f"kitten_generator_{args.variant}.mlpackage",
                default_bucket=256)

    return 0


if __name__ == "__main__":
    sys.exit(main())
