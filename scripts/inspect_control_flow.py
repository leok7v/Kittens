#!/usr/bin/env python3
"""Show Loop/If/Sequence nodes and all custom-domain nodes in detail.

Those are the most likely conversion blockers, so we dump them with their
subgraphs summarised so we can see what the hand-rolled STFT / SineGen
look like in the ONNX graph.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx


def summarise_graph(g, depth: int = 1) -> str:
    ind = "  " * depth
    lines = [f"{ind}inputs : {[i.name for i in g.input]}",
             f"{ind}outputs: {[o.name for o in g.output]}",
             f"{ind}nodes  : {len(g.node)}"]
    for n in g.node[:20]:
        lines.append(f"{ind}  {n.op_type:24s} {n.name or ''}  in={list(n.input)} out={list(n.output)}")
    if len(g.node) > 20:
        lines.append(f"{ind}  ... (+{len(g.node)-20} more)")
    return "\n".join(lines)


def main(path: Path) -> int:
    m = onnx.load(str(path))
    g = m.graph

    for i, n in enumerate(g.node):
        if n.op_type in {"Loop", "If", "Scan"}:
            print(f"[{i}] {n.op_type}  name={n.name!r}")
            print(f"     in  = {list(n.input)}")
            print(f"     out = {list(n.output)}")
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    print(f"     attr {a.name}: subgraph")
                    print(summarise_graph(a.g, depth=2))
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for j, sg in enumerate(a.graphs):
                        print(f"     attr {a.name}[{j}]: subgraph")
                        print(summarise_graph(sg, depth=2))
                else:
                    print(f"     attr {a.name}: type={a.type}")
            print()

    # Custom-domain nodes
    print("\n## Custom-domain nodes")
    for i, n in enumerate(g.node):
        if n.domain and n.domain not in ("", "ai.onnx"):
            print(f"[{i}] {n.op_type} domain={n.domain} name={n.name}")
            print(f"     in  = {list(n.input)}")
            print(f"     out = {list(n.output)}")
            for a in n.attribute:
                if a.type == onnx.AttributeProto.INT:
                    print(f"     attr {a.name}: {a.i}")
                elif a.type == onnx.AttributeProto.FLOAT:
                    print(f"     attr {a.name}: {a.f}")
                elif a.type == onnx.AttributeProto.STRING:
                    print(f"     attr {a.name}: {a.s!r}")
                elif a.type == onnx.AttributeProto.INTS:
                    print(f"     attr {a.name}: {list(a.ints)}")
                else:
                    print(f"     attr {a.name}: (type={a.type})")
            print()

    # Sequence ops
    print("\n## Sequence-op nodes")
    for i, n in enumerate(g.node):
        if "Sequence" in n.op_type or n.op_type in ("SplitToSequence", "ConcatFromSequence"):
            print(f"[{i}] {n.op_type}  {n.name or ''}  in={list(n.input)} out={list(n.output)}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?",
                    default="scripts/models/kitten_tts_nano_v0_8.onnx")
    args = ap.parse_args()
    sys.exit(main(Path(args.onnx)))
