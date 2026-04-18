#!/usr/bin/env python3
"""Run exposed ONNX and PyTorch port on the same input, compare intermediate
tensors. Used to find the first stage where our port diverges from ONNX.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import (  # noqa: E402
    KittenTTS, WeightBag, conv1d_ncl, conv_transpose_1d_ncl,
    reflection_pad_left, istft_head)
import torch.nn.functional as F


def run_onnx(path: Path, inputs: dict) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, inputs)
    return dict(zip(names, outs))


def compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        print(f"[{label}] shape mismatch: onnx={a.shape}  torch={b.shape}")
        # compare first n of common prefix
        n = min(a.size, b.size)
        a = a.flatten()[:n]
        b = b.flatten()[:n]
    else:
        a = a.flatten()
        b = b.flatten()
    a = a.astype(np.float64); b = b.astype(np.float64)
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    peak = float(np.max(np.abs(a - b)))
    num = float((a * b).sum())
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-30)
    cos = num / den
    print(f"[{label}] rms={rms:.4g} peak={peak:.4g} cos={cos:.6f} onnx_rms={np.sqrt((a**2).mean()):.4g} torch_rms={np.sqrt((b**2).mean()):.4g}")


def main():
    onnx_path = Path("scripts/models/kitten_exposed.onnx")
    safe = Path("Sources/KittenTTS/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    voices = np.load("scripts/models/voices.npz")

    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, 170, size=(1, 8), dtype=np.int64)
    style = voices["expr-voice-5-m"][8:9].astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    onnx_out = run_onnx(onnx_path, {"input_ids": input_ids, "style": style, "speed": speed})

    # Re-run PyTorch with intermediate hooks by instrumenting the forward.
    w = WeightBag.load(safe)
    model = KittenTTS(w).eval()
    torch.manual_seed(0)

    # Manually run the deterministic parts to grab f0_proj / n_proj / ups.1 etc.
    ids_t = torch.from_numpy(input_ids).long()
    style_t = torch.from_numpy(style).float()
    with torch.no_grad():
        B, L = ids_t.shape
        acoustic_style = style_t[:, :128]
        prosodic_style = style_t[:, 128:256]
        bert_out = model.bert(ids_t)
        prosody_in = bert_out @ model.beW + model.beB
        prosody = model.pred_text(prosody_in, prosodic_style)
        s_bcast = prosodic_style.reshape(B, 1, -1).expand(B, L, 128)
        prosody256 = torch.cat([prosody, s_bcast], dim=-1)
        prosody_ncl = prosody256.transpose(1, 2).contiguous()
        lstm_in = prosody_ncl.permute(2, 0, 1).contiguous()
        dy = model.duration_lstm(lstm_in)
        lstm_out = dy.permute(2, 0, 1, 3).reshape(1, L, 128)
        dur_logits = lstm_out @ model.dpW + model.dpB
        dur_sig = torch.sigmoid(dur_logits)
        dur_sum = dur_sig.sum(dim=-1)
        dur_scaled = dur_sum[0] / 1.0
        durs = torch.maximum(torch.tensor(1, dtype=torch.int32),
                             torch.round(dur_scaled).to(torch.int32))
        n_frames = int(durs.sum().item())
        align = torch.zeros(1, L, n_frames)
        j = 0
        for i, d in enumerate(durs.tolist()):
            align[0, i, j:j+int(d)] = 1.0
            j += int(d)
        prosody_lr_ncl = prosody_ncl @ align
        shared_in = prosody_lr_ncl.permute(2, 0, 1).contiguous()
        sy = model.shared_lstm(shared_in)
        fn_lstm_nlc = sy.permute(2, 0, 1, 3).reshape(1, n_frames, 128)
        fn_in_ncl = fn_lstm_nlc.transpose(1, 2).contiguous()

        f0 = model.f0_0(fn_in_ncl, prosodic_style, shortcut_input=fn_in_ncl)
        f0 = model.f0_1(f0, prosodic_style, shortcut_input=f0)
        f0 = model.f0_2(f0, prosodic_style, shortcut_input=f0)
        f0_proj = conv1d_ncl(f0, model.f0pW, model.f0pB)

        n = model.N_0(fn_in_ncl, prosodic_style, shortcut_input=fn_in_ncl)
        n = model.N_1(n, prosodic_style, shortcut_input=n)
        n = model.N_2(n, prosodic_style, shortcut_input=n)
        n_proj = conv1d_ncl(n, model.npW, model.npB)

        text_features_ncl = model.acoustic(ids_t)
        text_lr_ncl = text_features_ncl @ align
        dec_out = model.decoder(text_lr_ncl, f0_proj, n_proj, acoustic_style)

        # Generator manually to grab intermediates.
        from torch_kitten import compute_noise_contribs
        nr0, nr1 = compute_noise_contribs(
            f0_proj, n_frames, acoustic_style, w, model.noise_res0, model.noise_res1, seed=0)
        g = model.generator
        x = F.leaky_relu(dec_out, 0.1)
        x = conv_transpose_1d_ncl(x, g.u0W, g.u0B, stride=10, padding=5)
        x = x + nr0
        r0 = g.r0(x, acoustic_style)
        r1 = g.r1(x, acoustic_style)
        x = (r0 + r1) / 2.0
        x = F.leaky_relu(x, 0.1)
        ups1_out = conv_transpose_1d_ncl(x, g.u1W, g.u1B, stride=6, padding=3)
        refl_pad_out = reflection_pad_left(ups1_out, 1)
        x = refl_pad_out + nr1
        r2 = g.r2(x, acoustic_style)
        r3 = g.r3(x, acoustic_style)
        x = (r2 + r3) / 2.0
        x = F.leaky_relu(x, 0.1)
        conv_post_out = conv1d_ncl(x, g.cpW, g.cpB, padding=3)
        waveform = istft_head(conv_post_out, w)

    # Now compare against ONNX intermediates.
    print(f"durations onnx={onnx_out['duration'].tolist()}  torch={durs.tolist()}")
    compare("f0_proj", onnx_out["/F0_proj/Conv_output_0_Cast_to_float32_output_0"], f0_proj.numpy())
    compare("n_proj",  onnx_out["/N_proj/Conv_output_0_Cast_to_float32_output_0"], n_proj.numpy())
    compare("ups.1",   onnx_out["/decoder/generator/ups.1/ConvTranspose_output_0_Cast_to_float32_output_0"], ups1_out.numpy())
    compare("refl_pad", onnx_out["/decoder/generator/reflection_pad/Pad_output_0"], refl_pad_out.numpy())
    compare("conv_post", onnx_out["/decoder/generator/conv_post/Conv_output_0"], conv_post_out.numpy())
    compare("waveform", onnx_out["waveform"], waveform.numpy())


if __name__ == "__main__":
    main()
