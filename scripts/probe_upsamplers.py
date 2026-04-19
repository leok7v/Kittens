#!/usr/bin/env python3
"""Export just the two aggressive conv_transpose1d upsamplers (ups.0
stride=10 and ups.1 stride=6) as standalone mlpackages and feed identical
random inputs through torch + coreml to localize the generator bug."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import WeightBag  # noqa: E402

SR = 24000


def probe(mod: nn.Module, name: str, x: torch.Tensor) -> None:
    """Run torch + coreml on identical input and report spectral diff."""
    with torch.no_grad():
        ref = mod(x).numpy()
    print(f"\n==== {name} ====")
    print(f"  input shape  {tuple(x.shape)}")
    print(f"  torch output shape {ref.shape}")

    traced = torch.jit.trace(mod, x, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="y", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    cml = next(iter(ml.predict({"x": x.numpy()}).values()))
    print(f"  coreml output shape {cml.shape}")

    a = ref[0, 0].astype(np.float64) if ref.ndim == 3 else ref.flatten().astype(np.float64)
    b = cml[0, 0].astype(np.float64) if cml.ndim == 3 else cml.flatten().astype(np.float64)
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    err = b - a
    cos = float((a*b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-30))
    rms = float(np.sqrt((err**2).mean()))
    peak = float(np.max(np.abs(err)))
    tmax = float(np.max(np.abs(a)))
    print(f"  cos={cos:.6f}  rms_err={rms:.4g}  peak_err={peak:.4g}  torch_max={tmax:.4g}")
    if cos < 0.99:
        print(f"  *** DIVERGENCE ***")


class Ups0(nn.Module):
    """ups.0 conv_transpose1d — stride=10 upsample."""
    def __init__(self, bag: WeightBag):
        super().__init__()
        self.register_buffer("w", bag.f32("kmodel.decoder.generator.ups.0.weight"))
        self.register_buffer("b", bag.f32("kmodel.decoder.generator.ups.0.bias"))

    def forward(self, x):
        return F.conv_transpose1d(x, self.w, self.b, stride=10, padding=5)


class Ups1(nn.Module):
    """ups.1 conv_transpose1d — stride=6 upsample, then reflection pad left."""
    def __init__(self, bag: WeightBag):
        super().__init__()
        self.register_buffer("w", bag.f32("kmodel.decoder.generator.ups.1.weight"))
        self.register_buffer("b", bag.f32("kmodel.decoder.generator.ups.1.bias"))

    def forward(self, x):
        y = F.conv_transpose1d(x, self.w, self.b, stride=6, padding=3)
        # mirror torch_kitten.reflection_pad_left(y, 1)
        pref = y[..., 1:2].flip(-1)
        return torch.cat([pref, y], dim=-1)


def main():
    weights = Path("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    bag = WeightBag.load(weights)

    torch.manual_seed(0)

    # ups.0 takes (1, 256, N) where N is nFrames (up to 256 bucket)
    x0 = torch.randn(1, 256, 256, dtype=torch.float32) * 0.3
    probe(Ups0(bag).eval(), "ups.0 (stride=10)", x0)

    # ups.1 takes output of ups.0 = (1, 128, 2560) — actual shape from model
    # inspection, but exact in-channel count depends on model. Let's infer.
    ups1 = Ups1(bag).eval()
    in_ch = ups1.w.shape[0]
    print(f"  ups.1 in_channels (from weight) = {in_ch}")
    x1 = torch.randn(1, in_ch, 2560, dtype=torch.float32) * 0.3
    probe(ups1, "ups.1 (stride=6)", x1)


if __name__ == "__main__":
    main()
