#!/usr/bin/env python3
"""Focus on iSTFT head: feed the identical conv_post tensor through both
ONNX (via the exposed intermediates) and our Torch istft_head, and report
where the 2x magnitude difference comes from.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import WeightBag, istft_head


def main():
    path = Path("scripts/models/kitten_exposed.onnx")
    voices = np.load("scripts/models/voices.npz")
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, 170, size=(1, 8), dtype=np.int64)
    style = voices["expr-voice-5-m"][8:9].astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input_ids": input_ids, "style": style, "speed": speed})
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, outs))

    conv_post = out["/decoder/generator/conv_post/Conv_output_0"]
    mul = out["/decoder/generator/Mul_output_0"]       # exp * cos
    mul1 = out["/decoder/generator/Mul_1_output_0"]    # exp * sin
    ct = out["/decoder/generator/ConvTranspose_output_0"]   # conv_T(real)
    ct1 = out["/decoder/generator/ConvTranspose_1_output_0"]  # conv_T(imag)
    sub = out["/decoder/generator/Sub_1_output_0"]  # real - imag
    wav = out["waveform"]

    # Reproduce Mul / Mul_1 in torch from conv_post — should match exactly.
    cp_t = torch.from_numpy(conv_post).float()
    mag = torch.exp(cp_t[:, 0:11, :])
    phase = cp_t[:, 11:22, :]
    inner = torch.sin(phase)
    real = mag * torch.cos(inner)
    imag = mag * torch.sin(inner)

    def rpt(tag, a_np, b_torch):
        a = a_np.astype(np.float64).flatten()
        b = b_torch.numpy().astype(np.float64).flatten()
        n = min(a.size, b.size)
        rms = np.sqrt(((a[:n]-b[:n])**2).mean())
        peak = np.max(np.abs(a[:n]-b[:n]))
        cos = (a[:n]*b[:n]).sum() / (np.sqrt((a[:n]**2).sum()*(b[:n]**2).sum())+1e-30)
        print(f"{tag:30s} rms={rms:.4g}  peak={peak:.4g}  cos={cos:.6f}  "
              f"shapes onnx={a_np.shape} torch={tuple(b_torch.shape)}")
        print(f"{'':30s}   onnx rms={np.sqrt((a**2).mean()):.4g}  torch rms={np.sqrt((b**2).mean()):.4g}")

    rpt("Mul (exp*cos(sin(phase)))", mul, real)
    rpt("Mul_1 (exp*sin(sin(phase)))", mul1, imag)

    w = WeightBag.load("Sources/KittenTTS/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    wReal = w.f32("kmodel.decoder.generator.stft.weight_backward_real")
    wImag = w.f32("kmodel.decoder.generator.stft.weight_backward_imag")
    ar = F.conv_transpose1d(real, wReal, stride=5, padding=0)
    ai = F.conv_transpose1d(imag, wImag, stride=5, padding=0)
    rpt("ConvTranspose (real path)", ct, ar)
    rpt("ConvTranspose_1 (imag path)", ct1, ai)
    rpt("Sub_1 (ar - ai)", sub, (ar - ai))

    # Try alternative signs.
    rpt("Alt: ar + ai",  sub, (ar + ai))
    rpt("Alt: -ar + ai", sub, (-ar + ai))

    # Compare final waveform — apply the trim [10:-10].
    wav_t = istft_head(cp_t, w)
    rpt("waveform", wav, wav_t)
    print("first 16 samples onnx  =", wav[:16].tolist())
    print("first 16 samples torch =", wav_t[:16].tolist())

if __name__ == "__main__":
    main()
