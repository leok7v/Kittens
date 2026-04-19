#!/usr/bin/env python3
"""Validate the proposed fix for the cumsum-drift artifact.

Old: phase = (f0_harm / sr).cumsum(-1) * 2π over T_audio ≈ 144 k samples.
New: per-frame phase accumulation + within-frame linear extrapolation.
     short cumsum over T_frames ≈ 480, much less rounding drift.

If New matches Old bit-exactly in fp64 but differs in fp32 (and matches
torch's MLX reference more closely), we're on the right track.

Then we convert the New path to CoreML and check spectral diff vs torch.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

SR = 24000.0
HOP = 300  # samples per audio frame (= sr / 80Hz frame rate)
T_FRAMES = 480                       # 6 s at 80 fps
T_AUDIO = T_FRAMES * HOP             # 144 000 samples


class OldSineGen(nn.Module):
    """Reproduces the cumsum(-1) over T_audio sample path."""

    def forward(self, f0_audio, phase_jitter):
        harmonics = torch.arange(1, 10, dtype=torch.float32).reshape(1, 9, 1)
        f0_harm = f0_audio * harmonics          # (1, 9, T_audio)
        phase_inc = f0_harm / SR
        phase = phase_inc.cumsum(dim=-1) * (2.0 * math.pi)
        return torch.sin(phase + phase_jitter)


class NewSineGen(nn.Module):
    """Per-frame phase accumulation + linear within-frame.

    Given f0_per_frame (1, 1, T_frames), compute for each frame k and each
    within-frame sample t (0..HOP-1):

        phase(k, t) = phase_at_start(k) + f0_per_frame[k] * t * 2π / sr

    where phase_at_start(k) = 2π * cumsum(f0_per_frame * HOP / sr) up to k-1.

    Replaces a 144 000-element cumsum with a ≈480-element one.
    """

    def __init__(self) -> None:
        super().__init__()
        # (1, 1, 1, HOP) constant with values [0, 1, ..., HOP-1]
        self.register_buffer(
            "t_in_frame",
            torch.arange(HOP, dtype=torch.float32).reshape(1, 1, 1, HOP),
        )

    def forward(self, f0_per_frame, phase_jitter):
        # f0_per_frame: (1, 1, T_frames)
        harmonics = torch.arange(1, 10, dtype=torch.float32).reshape(1, 9, 1)
        f0_harm = f0_per_frame * harmonics                 # (1, 9, T_frames)

        step = f0_harm * (HOP / SR)                        # phase increment per frame (cycles)
        # phase at frame START, k=0 is zero (exclusive cumsum).
        phase_start = (step.cumsum(dim=-1) - step) * (2.0 * math.pi)  # (1,9,T_frames)

        # within-frame linear: f0_harm * t / SR * 2π (cycles within a frame)
        phase_within = (f0_harm.unsqueeze(-1) * self.t_in_frame) / SR * (2.0 * math.pi)
        phase = phase_start.unsqueeze(-1) + phase_within   # (1, 9, T_frames, HOP)
        phase = phase.reshape(1, 9, -1)                    # (1, 9, T_audio)
        return torch.sin(phase + phase_jitter)


def to_ml(mod, inputs, input_names):
    traced = torch.jit.trace(mod, inputs, strict=False)
    types = [ct.TensorType(name=n, shape=inp.shape, dtype=np.float32)
             for n, inp in zip(input_names, inputs)]
    return ct.convert(
        traced, inputs=types,
        outputs=[ct.TensorType(name="y", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )


def summary(tag, a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    err = b - a
    cos = float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-30))
    print(f"[{tag}] cos={cos:.6f}  rms={np.sqrt((err**2).mean()):.4g}  "
          f"peak={np.max(np.abs(err)):.4g}")


def main():
    torch.manual_seed(0)

    # Build a realistic f0 signal: piecewise constant per 300-sample frame.
    # Values in the range typical for Kiki (150-300 Hz).
    f0_per_frame = 150.0 + 100.0 * torch.rand(1, 1, T_FRAMES)   # (1,1,T_frames)
    f0_audio = f0_per_frame.repeat_interleave(HOP, dim=-1)      # (1,1,T_audio)
    phase_jitter = torch.rand(1, 9, 1) * 2.0 * math.pi

    old = OldSineGen().eval()
    new = NewSineGen().eval()

    # 1. torch: old vs new (should be equal in fp64, slightly different in fp32)
    with torch.no_grad():
        y_old = old(f0_audio, phase_jitter).numpy()
        y_new = new(f0_per_frame, phase_jitter).numpy()
    summary("torch  old-vs-new", y_old, y_new)

    # 2. coreml-old vs torch-old
    ml_old = to_ml(old, (f0_audio, phase_jitter), ["f0_audio", "phase_jitter"])
    y_cml_old = next(iter(ml_old.predict({
        "f0_audio": f0_audio.numpy(),
        "phase_jitter": phase_jitter.numpy(),
    }).values()))
    summary("coreml old-vs-torch old", y_cml_old, y_old)

    # 3. coreml-new vs torch-new  (should be near-zero error)
    ml_new = to_ml(new, (f0_per_frame, phase_jitter), ["f0_per_frame", "phase_jitter"])
    y_cml_new = next(iter(ml_new.predict({
        "f0_per_frame": f0_per_frame.numpy(),
        "phase_jitter": phase_jitter.numpy(),
    }).values()))
    summary("coreml new-vs-torch new", y_cml_new, y_new)

    # 4. coreml-new vs torch-old (want this close — drop-in replacement quality)
    summary("coreml new-vs-torch old", y_cml_new, y_old)


if __name__ == "__main__":
    main()
