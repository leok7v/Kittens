#!/usr/bin/env python3
"""Print a structural summary of the KittenTTS ONNX graph.

Purpose: enumerate inputs/outputs/initializers, op-type histogram, and
locate interesting nodes (iSTFT, Snake, custom ops) so we can plan
coremltools graph surgery and pick split points.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import onnx


def fmt_shape(tensor_type) -> str:
    dims = []
    for d in tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return "[" + ",".join(dims) + "]"


def elem_type_name(et: int) -> str:
    return onnx.TensorProto.DataType.Name(et)


def summarise(model_path: Path) -> int:
    model = onnx.load(str(model_path))
    g = model.graph

    print(f"# {model_path.name}")
    print(f"ir_version={model.ir_version}  opset={[o.version for o in model.opset_import]}")
    print(f"producer={model.producer_name} {model.producer_version}")
    print()

    print("## Inputs")
    for i in g.input:
        t = i.type.tensor_type
        print(f"  {i.name:40s} {elem_type_name(t.elem_type):8s} {fmt_shape(t)}")
    print()

    print("## Outputs")
    for o in g.output:
        t = o.type.tensor_type
        print(f"  {o.name:40s} {elem_type_name(t.elem_type):8s} {fmt_shape(t)}")
    print()

    # Op histogram
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"## Op histogram  ({len(g.node)} nodes, {len(ops)} distinct op types)")
    for op, n in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {op}")
    print()

    # Interesting / likely-unsupported ops
    interesting = {"STFT", "DFT", "ISTFT", "IDFT", "SequenceAt", "SequenceInsert",
                   "If", "Loop", "Scan", "Where", "NonZero", "Unique",
                   "DynamicQuantizeLinear", "QuantizeLinear", "DequantizeLinear",
                   "QLinearConv", "QLinearMatMul", "ConvInteger", "MatMulInteger"}
    print("## Nodes of interest (dynamic, custom, or quantization)")
    for i, n in enumerate(g.node):
        if n.op_type in interesting:
            ins = ", ".join(n.input)
            outs = ", ".join(n.output)
            print(f"  [{i:5d}] {n.op_type:24s} {n.name or '(anon)'}")
            print(f"         in  = {ins}")
            print(f"         out = {outs}")
    print()

    # Initializer stats
    total_bytes = 0
    dtype_hist: dict[int, int] = collections.Counter()
    for init in g.initializer:
        dtype_hist[init.data_type] += 1
        nb = len(init.raw_data) if init.raw_data else 0
        if not nb:
            # fall back to int_data / float_data etc. rough size
            nb = len(init.int32_data) * 4 + len(init.int64_data) * 8 + len(init.float_data) * 4
        total_bytes += nb
    print(f"## Initializers ({len(g.initializer)} tensors, ~{total_bytes/1024/1024:.1f} MiB)")
    for et, n in dtype_hist.items():
        print(f"  {elem_type_name(et):12s} {n}")
    print()

    # Scan for any Snake-activation-looking subgraph (sin + mul + div pattern
    # is too generic — just hint if the raw model seems to use custom ops).
    custom = [n for n in g.node if n.domain and n.domain not in ("", "ai.onnx")]
    if custom:
        print("## Custom-domain ops")
        for n in custom:
            print(f"  {n.op_type}  (domain={n.domain})  name={n.name}")
        print()

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?",
                    default="scripts/models/kitten_tts_nano_v0_8.onnx",
                    help="path to .onnx file")
    args = ap.parse_args()
    sys.exit(summarise(Path(args.onnx)))
