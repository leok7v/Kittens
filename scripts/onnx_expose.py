#!/usr/bin/env python3
"""Modify the ONNX graph to expose named intermediate tensors as outputs.

Usage:
    python onnx_expose.py SRC DST tensor1 tensor2 ...

The destination ONNX runs the same computation but emits the requested
tensors alongside the original outputs, so we can compare PyTorch/MLX ports
against the reference at bisection points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx


def expose(src: Path, dst: Path, names: list[str]) -> None:
    m = onnx.load(str(src))
    g = m.graph
    existing = {o.name for o in g.output}
    # All intermediate tensor names produced by some node.
    produced: dict[str, onnx.NodeProto] = {}
    for n in g.node:
        for out in n.output:
            produced[out] = n

    missing = [n for n in names if n not in produced and n not in existing]
    if missing:
        print(f"error: these names are not produced by any node: {missing}",
              file=sys.stderr)
        # Dump candidates that contain any of the missing substrings for help.
        for needle in missing:
            cands = [p for p in produced if needle.split("/")[-1] in p][:20]
            if cands:
                print(f"  maybe you meant one of: {cands}", file=sys.stderr)
        sys.exit(2)

    # Try to infer dtypes by running shape inference once.
    try:
        inferred = onnx.shape_inference.infer_shapes(m)
        by_name = {vi.name: vi for vi in inferred.graph.value_info}
    except Exception:
        by_name = {}

    for name in names:
        if name in existing:
            continue
        vi = by_name.get(name)
        if vi is not None:
            g.output.append(vi)
        else:
            # Fall back to FLOAT; we only expose float tensors for numerical diff.
            g.output.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, None))

    onnx.save(m, str(dst))
    print(f"saved {dst} with {len(g.output)} outputs")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    expose(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
