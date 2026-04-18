#!/usr/bin/env python3
"""Attempt a direct coremltools conversion of the untouched ONNX.

Expected to fail — point is to capture the exact error so we know which
graph-surgery steps are actually required. Errors get written next to
the model; ct.convert prints a traceback on failure.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import onnx
import coremltools as ct
from coremltools.converters.mil.frontend.milproto import load as _milproto_load  # noqa: F401


def main() -> int:
    onnx_path = Path("scripts/models/kitten_tts_nano_v0_8.onnx")
    model = onnx.load(str(onnx_path))
    print(f"opset: {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")

    try:
        mlmodel = ct.converters.onnx.convert(model)
        out = Path("scripts/models/kitten_naive.mlpackage")
        mlmodel.save(str(out))
        print(f"saved: {out}")
        return 0
    except AttributeError as e:
        print(f"ct.converters.onnx not in coremltools 9: {e}")
    except Exception as e:
        print(f"other error (onnx path): {e}")

    # coremltools 7+ path: convert via source="pytorch" requires a torch model.
    # For ONNX → MIL, the supported path is: load ONNX, feed through
    # onnxruntime.tools.symbolic_shape_infer → coremltools? Actually, in ct 7+
    # they deprecated the direct ONNX path. Canonical flow: convert to TorchScript
    # via onnx2pytorch or use ct.convert(onnx_model, source="onnx").
    # Try that form:
    try:
        mlmodel = ct.convert(str(onnx_path),
                             source="onnx",
                             convert_to="mlprogram",
                             minimum_deployment_target=ct.target.macOS15)
        out = Path("scripts/models/kitten_naive.mlpackage")
        mlmodel.save(str(out))
        print(f"saved: {out}")
        return 0
    except Exception:
        print("ct.convert(source='onnx') failed with:")
        traceback.print_exc()

    return 1


if __name__ == "__main__":
    sys.exit(main())
