#!/usr/bin/env python3
"""Test whether backward-LSTM pad leakage is the cause of CoreML speech weirdness.

Compare three things for the SAME text + voice:
  (1) Torch monolithic (equivalent to MLX): runs at actual L, no padding
      anywhere. This is our reference.
  (2) Torch stage at bucketed L (e.g. L=16): pads to 16, uses the mask.
      Outputs should agree with (1) if the mask is perfect.
  (3) CoreML TextStage at bucketed L=16. Should match (2) if conversion
      preserves semantics.

If (2) diverges from (1), the mask is not enough — likely LSTM bias drift
through pad positions. If (3) diverges from (2), the conversion isn't
faithful. Both tests at a single TextStage output (dur_sig), since that's
what drives the duration errors the user reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import (TextStage, KittenTTS, WeightBag)  # noqa: E402

import coremltools as ct  # noqa: E402

SAFE = "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
VOICES = "scripts/models/voices.npz"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=12)
    ap.add_argument("--bucket", type=int, default=16)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    real_L = args.real
    bucket_L = args.bucket
    ids_real = rng.integers(1, 170, size=(1, real_L), dtype=np.int64)
    ids_pad = np.zeros((1, bucket_L), dtype=np.int64)
    ids_pad[0, :real_L] = ids_real[0]
    mask = np.zeros((1, bucket_L), dtype=np.float32)
    mask[0, :real_L] = 1.0
    style = np.load(VOICES)["expr-voice-5-m"][real_L:real_L + 1].astype(np.float32)

    w = WeightBag.load(SAFE)

    # (1) Torch TextStage at actual L (no padding) — using the monolithic KittenTTS
    # would also work, but we just need the dur_sig which TextStage exposes.
    stage_actual = TextStage(w).eval()
    with torch.no_grad():
        mask_allones = torch.ones(1, real_L, dtype=torch.float32)
        p_r, tf_r, ds_r = stage_actual(torch.from_numpy(ids_real).long(),
                                       torch.from_numpy(style),
                                       mask_allones)
    # durations from ds_r
    d_ref = ds_r[0].sum(dim=-1).numpy()  # (real_L,)

    # (2) Torch TextStage padded to bucket_L with mask
    with torch.no_grad():
        p_p, tf_p, ds_p = stage_actual(torch.from_numpy(ids_pad).long(),
                                       torch.from_numpy(style),
                                       torch.from_numpy(mask))
    d_pad = ds_p[0, :real_L].sum(dim=-1).numpy()

    # (3) CoreML TextStage at bucket_L
    ml = ct.models.MLModel(f"scripts/models/kitten_text_L{bucket_L}.mlpackage")
    out = ml.predict({
        "input_ids": ids_pad.astype(np.int32),
        "style": style,
        "attention_mask": mask,
    })
    ds_ml = None
    for v in out.values():
        if v.shape == (1, bucket_L, 50):
            ds_ml = v
            break
    d_ml = ds_ml[0, :real_L].sum(axis=-1)

    print(f"Real phonemes: {real_L}   bucket: {bucket_L}   pad: {bucket_L - real_L}")
    print(f"\nPer-phoneme duration sum (smaller sum == shorter phoneme):")
    print(f"  (1) torch no-pad  : {[f'{x:.2f}' for x in d_ref.tolist()]}")
    print(f"  (2) torch padded  : {[f'{x:.2f}' for x in d_pad.tolist()]}")
    print(f"  (3) coreml padded : {[f'{x:.2f}' for x in d_ml.tolist()]}")

    diff_12 = d_ref - d_pad
    diff_13 = d_ref - d_ml
    diff_23 = d_pad - d_ml
    def stats(name, d):
        print(f"  {name}: max-abs = {float(np.abs(d).max()):.3f}   rms = {float(np.sqrt((d**2).mean())):.3f}")
    print(f"\nAbsolute diffs:")
    stats("(1)-(2) no-pad vs torch-padded  ", diff_12)
    stats("(1)-(3) no-pad vs coreml-padded ", diff_13)
    stats("(2)-(3) torch-padded vs coreml  ", diff_23)


if __name__ == "__main__":
    main()
