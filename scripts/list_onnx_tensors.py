#!/usr/bin/env python3
"""List intermediate ONNX tensor names matching a pattern."""
import re, sys
from pathlib import Path
import onnx

def main():
    path = Path("scripts/models/kitten_tts_nano_v0_8.onnx")
    pat = re.compile(sys.argv[1]) if len(sys.argv) > 1 else re.compile(".")
    m = onnx.load(str(path))
    for n in m.graph.node:
        for out in n.output:
            if pat.search(out):
                print(f"{n.op_type:24s} {out}")

if __name__ == "__main__":
    main()
