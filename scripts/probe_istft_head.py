#!/usr/bin/env python3
"""Export just the iSTFT head (conv_transpose real/imag + trim) as its own
mlpackage, then feed identical synthetic conv_post tensors through torch
and CoreML. Compare outputs bin-by-bin and spectrally.

Goal: prove (or rule out) istft_head as the source of the CoreML tonal
artifact around ~720 Hz we observed vs torch/MLX.
"""
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


class ISTFTHead(nn.Module):
    """Matches torch_kitten.istft_head exactly. Input shape (1, 22, T)."""

    def __init__(self, w: WeightBag, trim: int = 10):
        super().__init__()
        self.trim = trim
        wReal = w.f32("kmodel.decoder.generator.stft.weight_backward_real")
        wImag = w.f32("kmodel.decoder.generator.stft.weight_backward_imag")
        self.register_buffer("wReal", wReal)
        self.register_buffer("wImag", wImag)

    def forward(self, conv_post_out: torch.Tensor) -> torch.Tensor:
        mag_logits = conv_post_out[:, 0:11, :]
        phase = conv_post_out[:, 11:22, :]
        mag = torch.exp(mag_logits)
        inner = torch.sin(phase)
        real = mag * torch.cos(inner)
        imag = mag * torch.sin(inner)
        audio_real = F.conv_transpose1d(real, self.wReal, stride=5, padding=0)
        audio_imag = F.conv_transpose1d(imag, self.wImag, stride=5, padding=0)
        audio = audio_real - audio_imag
        # Fixed trim slice; T is the static output length so we can hard-
        # code the end index. coremltools' aten::Int rewrite of `T - trim`
        # fails otherwise.
        return audio[:, :, self.trim : -self.trim]


def main():
    weights = Path("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    bag = WeightBag.load(weights)
    head = ISTFTHead(bag).eval()
    print(f"iSTFT weight shapes:")
    print(f"  weight_backward_real {tuple(head.wReal.shape)}")
    print(f"  weight_backward_imag {tuple(head.wImag.shape)}")

    # Synthetic input shaped like a typical GeneratorPipeline conv_post output.
    # Use a fixed T that mirrors what we saw in verify_coreml_vs_torch.py runs.
    T = 307   # arbitrary; matches (nBucket * 600 - trim_padding) / stride
    torch.manual_seed(0)
    x = torch.randn(1, 22, T, dtype=torch.float32) * 0.3

    with torch.no_grad():
        ref = head(x).numpy()[0, 0]   # (T_out,)
    print(f"torch istft output: len={len(ref)}")

    # Trace and convert. Keep a fixed T so we don't need dynamic shapes.
    traced = torch.jit.trace(head, x, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 22, T), dtype=np.float32)],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    cml_out = ml.predict({"x": x.numpy()})
    # Output shape comes out (1, 1, T_out)
    ml_audio = next(iter(cml_out.values())).reshape(-1)
    print(f"coreml istft output: len={len(ml_audio)}")

    # Compare
    n = min(len(ref), len(ml_audio))
    a = ref[:n]; b = ml_audio[:n]
    diff = b - a
    cos = float((a*b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-30))
    rms = float(np.sqrt((diff**2).mean()))
    peak = float(np.max(np.abs(diff)))
    print(f"\n[istft_head] cos={cos:.6f} rms={rms:.4g} peak={peak:.4g} "
          f"ref_rms={float(np.sqrt((a**2).mean())):.4g} test_rms={float(np.sqrt((b**2).mean())):.4g}")

    # Spectral peak of the error.
    spec_err = np.abs(np.fft.rfft(diff * np.hanning(len(diff))))
    spec_ref = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(diff), 1.0 / SR)
    # Top 8 peaks in error where freq > 50 Hz.
    ranked = np.argsort(spec_err)[::-1]
    print(f"\ntop error-spectrum peaks:")
    print(f"  {'freq_Hz':>10} {'err_mag':>10} {'ref_mag':>10} {'err/ref_dB':>12}")
    printed = 0; last_f = -1
    for idx in ranked:
        f = freqs[idx]
        if f < 50: continue
        if abs(f - last_f) < 50: continue
        last_f = f
        e = spec_err[idx]; r = spec_ref[idx]
        db = 20 * np.log10((e + 1e-9) / (r + 1e-9))
        print(f"  {f:>10.1f} {e:>10.3f} {r:>10.3f} {db:>+12.1f}")
        printed += 1
        if printed >= 8: break


if __name__ == "__main__":
    main()
