#!/usr/bin/env python3
"""Probe candidate ops in the noise/sine-gen path — cumsum over a long
axis is a known coremltools precision hotspot, and atan2 rewrites can
drift.

Feeds deterministic synthetic inputs through torch + coreml versions of:
  A. `phase_inc.cumsum(dim=-1)` over a length-matched tensor.
  B. The sine-gen block (cumsum + phase_jitter + sin).
  C. atan2 alone.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

SR = 24000
T_AUDIO = 24_000 * 6  # 6 s of audio at 24 kHz — roughly full-sentence length


class CumsumOnly(nn.Module):
    def forward(self, x):
        return x.cumsum(dim=-1)


class SineGen(nn.Module):
    def forward(self, phase_inc, phase_jitter):
        phase = phase_inc.cumsum(dim=-1) * (2.0 * math.pi)
        return torch.sin(phase + phase_jitter)


class Atan2(nn.Module):
    def forward(self, y, x):
        return torch.atan2(y, x)


def to_ml(mod, inputs, input_names):
    traced = torch.jit.trace(mod, inputs, strict=False)
    types = [ct.TensorType(name=n, shape=inp.shape, dtype=np.float32)
             for n, inp in zip(input_names, inputs)]
    ml = ct.convert(
        traced, inputs=types,
        outputs=[ct.TensorType(name="y", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    return ml


def compare(label, a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    err = b - a
    cos = float((a*b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-30))
    rms = float(np.sqrt((err**2).mean()))
    peak = float(np.max(np.abs(err)))
    print(f"[{label}] cos={cos:.6f}  rms={rms:.4g}  peak={peak:.4g}  "
          f"ref_max={float(np.max(np.abs(a))):.4g}")


def main():
    torch.manual_seed(0)

    # ---- A. cumsum on long axis ----
    # Same per-sample magnitudes as the noise path: phase_inc ≈ f0/sr ≈ 0.01.
    x = torch.rand(1, 9, T_AUDIO) * 0.01
    mod = CumsumOnly().eval()
    ml = to_ml(mod, (x,), ["x"])
    with torch.no_grad():
        ref = mod(x).numpy()
    cml = next(iter(ml.predict({"x": x.numpy()}).values()))
    compare("cumsum", ref, cml)

    # ---- B. sine-gen ----
    phase_inc = torch.rand(1, 9, T_AUDIO) * 0.01
    phase_jitter = torch.rand(1, 9, 1) * 2 * math.pi
    mod = SineGen().eval()
    ml = to_ml(mod, (phase_inc, phase_jitter), ["phase_inc", "phase_jitter"])
    with torch.no_grad():
        ref = mod(phase_inc, phase_jitter).numpy()
    cml = next(iter(ml.predict({
        "phase_inc": phase_inc.numpy(),
        "phase_jitter": phase_jitter.numpy(),
    }).values()))
    compare("sine-gen", ref, cml)

    # ---- C. atan2 ----
    yy = torch.randn(1, 11, T_AUDIO // 5) * 0.5
    xx = torch.randn(1, 11, T_AUDIO // 5) * 0.5
    mod = Atan2().eval()
    ml = to_ml(mod, (yy, xx), ["yy", "xx"])
    with torch.no_grad():
        ref = mod(yy, xx).numpy()
    cml = next(iter(ml.predict({"yy": yy.numpy(), "xx": xx.numpy()}).values()))
    compare("atan2", ref, cml)


if __name__ == "__main__":
    main()
