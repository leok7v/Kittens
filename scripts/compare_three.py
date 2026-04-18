#!/usr/bin/env python3
"""Given the exact phoneme IDs + voice + speed from a failing Swift run,
compare four duration sources:
  1. ONNX Runtime on kitten_tts_nano_v0_8.onnx (the reference)
  2. torch KittenTTS (monolithic, no-pad, matches MLX in structure)
  3. torch TextStage (padded, with attention_mask)
  4. CoreML TextStage (padded mlpackage)

Finds where the implementations diverge.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import KittenTTS, TextStage, WeightBag  # noqa: E402
import coremltools as ct  # noqa: E402

SAFE = "Sources/KittenTTS/Resources/nano/kitten_tts_nano_v0_8.safetensors"
ONNX = "scripts/models/kitten_tts_nano_v0_8.onnx"


# Hardcoded from Swift debug output for "Kitten TTS is now streaming audio chunks for lower latency,"
PHONEMES = [0, 53, 156, 102, 148, 56, 175, 16, 62, 157, 51, 158, 62, 157, 51, 158, 156, 86,
            61, 16, 102, 68, 16, 56, 156, 43, 135, 16, 61, 62, 123, 156, 51, 158, 55, 102,
            112, 16, 156, 76, 158, 46, 102, 157, 57, 135, 16, 62, 131, 156, 138, 112, 53,
            61, 16, 48, 76, 158, 123, 16, 54, 156, 57, 135, 85, 16, 54, 156, 47, 102, 62,
            83, 56, 61, 51, 3, 10, 0]
STYLE_FIRST4 = [0.38681412, 0.043642446, 0.005097732, 0.055575598]
SPEED = 0.8


def get_style() -> np.ndarray:
    # Swift reads row refId=59 of voices.safetensors / voices.npz.
    # Use voices.npz for clarity.
    v = np.load("scripts/models/voices.npz")
    style = v["expr-voice-5-m"][59:60].astype(np.float32)
    # Verify first 4 values match the Swift dump.
    for a, b in zip(style[0, :4].tolist(), STYLE_FIRST4):
        assert abs(a - b) < 1e-5, f"style mismatch {a} vs {b}"
    return style


def onnx_durations(phonemes, style, speed):
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    ids = np.array(phonemes, dtype=np.int64).reshape(1, -1)
    outs = sess.run(None, {
        "input_ids": ids,
        "style": style,
        "speed": np.array([speed], dtype=np.float32),
    })
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, outs))
    return out["duration"]


def torch_mono_durations(phonemes, style, speed):
    w = WeightBag.load(SAFE)
    model = KittenTTS(w).eval()
    ids = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0)
    style_t = torch.from_numpy(style)
    with torch.no_grad():
        _, durs = model(ids, style_t, speed=speed, noise_seed=0)
    return durs.cpu().numpy().astype(np.int64)


def torch_stage_durations(phonemes, style, speed, L):
    w = WeightBag.load(SAFE)
    stage = TextStage(w).eval()
    realL = len(phonemes)
    ids_np = np.zeros((1, L), dtype=np.int64)
    ids_np[0, :realL] = phonemes
    mask_np = np.zeros((1, L), dtype=np.float32)
    mask_np[0, :realL] = 1.0
    with torch.no_grad():
        p, t, ds = stage(torch.from_numpy(ids_np).long(),
                         torch.from_numpy(style),
                         torch.from_numpy(mask_np))
    durs_sum = ds[0, :realL].sum(dim=-1).numpy()
    return np.maximum(1, np.round(durs_sum / speed).astype(np.int32))


def coreml_durations(phonemes, style, speed, L):
    m = ct.models.MLModel(f"scripts/models/kitten_text_L{L}.mlpackage")
    realL = len(phonemes)
    ids_np = np.zeros((1, L), dtype=np.int32)
    ids_np[0, :realL] = phonemes
    mask_np = np.zeros((1, L), dtype=np.float32)
    mask_np[0, :realL] = 1.0
    out = m.predict({"input_ids": ids_np, "style": style, "attention_mask": mask_np})
    ds = None
    for v in out.values():
        if v.shape == (1, L, 50):
            ds = v; break
    assert ds is not None
    durs_sum = ds[0, :realL].sum(axis=-1)
    return np.maximum(1, np.round(durs_sum / speed).astype(np.int32))


def main():
    style = get_style()
    phonemes = PHONEMES
    L_bucket = 128  # matches Swift's pick for 78-phoneme input
    print(f"Phonemes ({len(phonemes)}): {phonemes[:10]} ...")

    d_onnx = onnx_durations(phonemes, style, SPEED)
    d_torch_mono = torch_mono_durations(phonemes, style, SPEED)
    d_torch_stage = torch_stage_durations(phonemes, style, SPEED, L_bucket)
    d_coreml = coreml_durations(phonemes, style, SPEED, L_bucket)

    print(f"\nONNX  nFrames={int(d_onnx.sum())}   durs={d_onnx.tolist()}")
    print(f"\nTorch monolithic (no pad)")
    print(f"      nFrames={int(d_torch_mono.sum())}   durs={d_torch_mono.tolist()}")
    print(f"\nTorch TextStage padded L={L_bucket}")
    print(f"      nFrames={int(d_torch_stage.sum())}   durs={d_torch_stage.tolist()}")
    print(f"\nCoreML TextStage L={L_bucket}")
    print(f"      nFrames={int(d_coreml.sum())}   durs={d_coreml.tolist()}")

    def cmp(a_name, a, b_name, b):
        diff = np.abs(a - b).max()
        print(f"  {a_name} vs {b_name}: max-abs diff = {int(diff)}")

    print("\nPairwise:")
    cmp("ONNX", d_onnx, "TorchMono", d_torch_mono)
    cmp("TorchMono", d_torch_mono, "TorchStagePad", d_torch_stage)
    cmp("TorchStagePad", d_torch_stage, "CoreML", d_coreml)
    cmp("ONNX", d_onnx, "CoreML", d_coreml)


if __name__ == "__main__":
    main()
