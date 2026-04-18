#!/usr/bin/env python3
"""Trace TextStage / GeneratorStage with sample inputs and run coremltools.

Strategy:
  * Trace at one representative shape first to prove the graph converts.
  * Then enumerate shape buckets (L ∈ {16, 32, 64, 128} etc.) to avoid the
    full dynamic-shape conversion cost.

Run subcommand(s):
    python convert_to_coreml.py text         # convert TextStage only
    python convert_to_coreml.py generator    # convert GeneratorStage only
    python convert_to_coreml.py both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import TextStage, GeneratorStage, WeightBag  # noqa: E402

import coremltools as ct  # noqa: E402


def _patch_coremltools_int_cast():
    """coremltools 9 _int handler crashes when torch's jit emits aten::Int on
    a single-element array instead of a scalar (seen inside nn.LSTM traces).
    Wrap dtype(x.val) to flatten single-element arrays to scalars first."""
    from coremltools.converters.mil.frontend.torch import ops as _tops
    from coremltools.converters.mil.mil import Builder as _mb
    import numpy as _np

    orig_cast = _tops._cast

    def patched_cast(context, node, dtype, dtype_str):
        x = context[node.inputs[0]]
        if hasattr(x, "val") and x.val is not None and hasattr(x.val, "flatten"):
            arr = _np.asarray(x.val)
            if arr.ndim > 0 and arr.size == 1:
                res = _mb.const(val=dtype(arr.flatten()[0]), name=node.name)
                context.add(res)
                return
        return orig_cast(context, node, dtype, dtype_str)

    _tops._cast = patched_cast


_patch_coremltools_int_cast()


OUT_DIR = Path("scripts/models")


def _sample_style() -> torch.Tensor:
    v = np.load("scripts/models/voices.npz")
    return torch.from_numpy(v["expr-voice-5-m"][32:33].astype(np.float32))


def convert_text_stage(L: int = 32, save: bool = True,
                       flex: bool = False) -> None:
    tag = f"flex_L{L}" if flex else f"L{L}"
    print(f"=== TextStage {tag} ===")
    w = WeightBag.load("Sources/KittenTTS/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    stage = TextStage(w).eval()

    input_ids = torch.randint(1, 170, (1, L), dtype=torch.long)
    style = _sample_style()

    with torch.no_grad():
        p, t, d = stage(input_ids, style)
    print(f"  torch outputs: prosody_ncl={tuple(p.shape)} text_features={tuple(t.shape)} dur_sig={tuple(d.shape)}")

    print("  tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(stage, (input_ids, style), check_trace=False, strict=False)

    print(f"  converting to Core ML (flex={flex}) ...")
    if flex:
        ids_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=4, upper_bound=512, default=L)))
    else:
        ids_shape = input_ids.shape
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=ids_shape, dtype=np.int32),
            ct.TensorType(name="style", shape=style.shape, dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.short_description = f"KittenTTS text stage ({tag})"
    if save:
        out = OUT_DIR / f"kitten_text_{tag}.mlpackage"
        mlmodel.save(str(out))
        print(f"  saved {out}")
    print()


def convert_generator_stage(n_frames: int = 32, save: bool = True,
                            flex: bool = False) -> None:
    tag = f"flex_N{n_frames}" if flex else f"N{n_frames}"
    print(f"=== GeneratorStage {tag} ===")
    w = WeightBag.load("Sources/KittenTTS/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    stage = GeneratorStage(w).eval()

    prosody_lr = torch.randn(1, 256, n_frames)
    text_lr = torch.randn(1, 128, n_frames)
    style = _sample_style()

    with torch.no_grad():
        wav = stage(prosody_lr, text_lr, style)
    print(f"  torch output: waveform={tuple(wav.shape)}")

    print("  tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(stage, (prosody_lr, text_lr, style),
                                 check_trace=False, strict=False)

    print(f"  converting to Core ML (flex={flex}) ...")
    if flex:
        nf_dim = ct.RangeDim(lower_bound=16, upper_bound=1024, default=n_frames)
        p_shape = ct.Shape(shape=(1, 256, nf_dim))
        t_shape = ct.Shape(shape=(1, 128, nf_dim))
    else:
        p_shape = prosody_lr.shape
        t_shape = text_lr.shape
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="prosody_lr", shape=p_shape, dtype=np.float32),
            ct.TensorType(name="text_lr",    shape=t_shape, dtype=np.float32),
            ct.TensorType(name="style",      shape=style.shape, dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.short_description = f"KittenTTS generator stage ({tag})"
    if save:
        out = OUT_DIR / f"kitten_generator_{tag}.mlpackage"
        mlmodel.save(str(out))
        print(f"  saved {out}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["text", "generator", "both"],
                    default="both", nargs="?")
    ap.add_argument("-L", type=int, default=32, help="L for TextStage bucket")
    ap.add_argument("-N", type=int, default=32, help="nFrames for GeneratorStage bucket")
    ap.add_argument("--flex", action="store_true",
                    help="emit RangeDim flex shapes (L ∈ [4,512], N ∈ [16,1024])")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.which in ("text", "both"):
        convert_text_stage(args.L, save=not args.no_save, flex=args.flex)
    if args.which in ("generator", "both"):
        convert_generator_stage(args.N, save=not args.no_save, flex=args.flex)


if __name__ == "__main__":
    main()
