#!/usr/bin/env python3
"""Quick parity check: INT8-quantized TextStage vs fp32 torch reference.

Duration predictions are the most sensitive thing the text stage produces;
if INT8 drifts them significantly, audio is audibly different. Uses the
real-phoneme set we already verified end-to-end so we can compare apples
to apples.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import TextStage, WeightBag  # noqa: E402
import coremltools as ct  # noqa: E402

SAFE = "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
FP32 = "Sources/KittenApp/Resources/coreml/kitten_text_L{L}.mlpackage"
INT8 = "scripts/models/quantized/kitten_text_L{L}.mlpackage"
VOICES = "scripts/models/voices.npz"

# Real-world phonemes from "Kitten TTS is now streaming audio chunks for lower latency,"
PHONEMES = [0, 53, 156, 102, 148, 56, 175, 16, 62, 157, 51, 158, 62, 157, 51, 158, 156, 86,
            61, 16, 102, 68, 16, 56, 156, 43, 135, 16, 61, 62, 123, 156, 51, 158, 55, 102,
            112, 16, 156, 76, 158, 46, 102, 157, 57, 135, 16, 62, 131, 156, 138, 112, 53,
            61, 16, 48, 76, 158, 123, 16, 54, 156, 57, 135, 85, 16, 54, 156, 47, 102, 62,
            83, 56, 61, 51, 3, 10, 0]


def coreml_durs(path: str, phonemes, style, speed, L: int) -> np.ndarray:
    m = ct.models.MLModel(path)
    realL = len(phonemes)
    ids = np.zeros((1, L), dtype=np.int32)
    ids[0, :realL] = phonemes
    mask = np.zeros((1, L), dtype=np.float32)
    mask[0, :realL] = 1.0
    out = m.predict({"input_ids": ids, "style": style,
                     "attention_mask": mask})
    ds = next(v for v in out.values() if v.shape == (1, L, 50))
    return np.maximum(1, np.round(ds[0, :realL].sum(axis=-1) / speed).astype(np.int32))


def torch_durs(phonemes, style, speed, L: int) -> np.ndarray:
    w = WeightBag.load(SAFE)
    stage = TextStage(w).eval()
    realL = len(phonemes)
    ids = np.zeros((1, L), dtype=np.int64)
    ids[0, :realL] = phonemes
    mask = np.zeros((1, L), dtype=np.float32)
    mask[0, :realL] = 1.0
    with torch.no_grad():
        _, _, ds = stage(torch.from_numpy(ids).long(),
                         torch.from_numpy(style),
                         torch.from_numpy(mask))
    dursum = ds[0, :realL].sum(dim=-1).numpy()
    return np.maximum(1, np.round(dursum / speed).astype(np.int32))


def main():
    # Use a bucket that's big enough to hold all the phonemes we'll exercise.
    # PHONEMES has 78; needs L >= 128.
    L = 128
    speed = 0.8
    style = np.load(VOICES)["expr-voice-5-m"][59:60].astype(np.float32)

    short = PHONEMES[:12]  # exercises small pad-count case too

    for label, ph in [("full (78 phonemes)", PHONEMES), ("short (12 phonemes)", short)]:
        L_bucket = L if len(ph) > 64 else 64
        d_ref = torch_durs(ph, style, speed, L_bucket)
        d_fp32 = coreml_durs(FP32.format(L=L_bucket), ph, style, speed, L_bucket)
        d_int8 = coreml_durs(INT8.format(L=L_bucket), ph, style, speed, L_bucket)

        print(f"\n{label}  (L bucket = {L_bucket})")
        print(f"  torch ref  nFrames={int(d_ref.sum())}  durs={d_ref.tolist()}")
        print(f"  CoreML fp32 nFrames={int(d_fp32.sum())}  durs={d_fp32.tolist()}")
        print(f"  CoreML INT8 nFrames={int(d_int8.sum())}  durs={d_int8.tolist()}")
        def cmp(a, b):
            return int(np.abs(a - b).max()), int(np.abs(a - b).sum())
        m12, s12 = cmp(d_ref, d_fp32)
        m13, s13 = cmp(d_ref, d_int8)
        m23, s23 = cmp(d_fp32, d_int8)
        print(f"  ref↔fp32  max={m12}  sum|Δ|={s12}")
        print(f"  ref↔INT8  max={m13}  sum|Δ|={s13}")
        print(f"  fp32↔INT8 max={m23}  sum|Δ|={s23}")


if __name__ == "__main__":
    main()
