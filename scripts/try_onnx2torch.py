#!/usr/bin/env python3
"""Attempt ONNX → torch.nn.Module via onnx2torch.

If this succeeds, we get:
  (1) a PyTorch model that coremltools can trace and convert, and
  (2) a parallel numerical reference for Python-side golden checks.

The conversion is expected to choke on:
  - DynamicQuantizeLSTM (com.microsoft custom op)
  - ConvInteger / MatMulInteger (quantized int8 ops)
  - Loop / SequenceEmpty / SequenceInsert / ConcatFromSequence (length regulation)

Capture the exact failures so we can decide between:
  (a) dequantize + strip Loop in ONNX before onnx2torch, or
  (b) write the forward pass in PyTorch from scratch using weights from the
      existing .safetensors file (mirrors what TTS.swift already does in MLX).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import onnx
import torch
from onnx2torch import convert


def main() -> int:
    path = Path("scripts/models/kitten_tts_nano_v0_8.onnx")
    model = onnx.load(str(path))
    print(f"opsets: {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print(f"nodes:  {len(model.graph.node)}")
    try:
        torch_module = convert(model)
    except Exception:
        print("onnx2torch.convert failed:")
        traceback.print_exc()
        return 1

    print("onnx2torch.convert succeeded")
    print(torch_module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
