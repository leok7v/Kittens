#!/usr/bin/env python3
"""Run torch_kitten.py end-to-end on a sentence and save a .wav.

Used to A/B the PyTorch reference against the MLX and CoreML backends —
if MLX ≈ Torch but CoreML diverges (or vice versa), we've isolated the
source of numerical drift.

Usage:
    python scripts/torch_generate.py --text "Hello world." --voice Kiki \
        --out tmp/torch.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from torch_kitten import KittenTTS, WeightBag
from real_calibration import VOICES_NPZ, WEIGHTS, build_vocab, phonemize

VOICE_ALIASES = {
    "Bella": "expr-voice-2-f", "Jasper": "expr-voice-2-m",
    "Luna":  "expr-voice-3-f", "Bruno":  "expr-voice-3-m",
    "Rosie": "expr-voice-4-f", "Hugo":   "expr-voice-4-m",
    "Kiki":  "expr-voice-5-f", "Leo":    "expr-voice-5-m",
}
SPEED_PRIORS = {
    "expr-voice-2-f": 0.8, "expr-voice-2-m": 0.8,
    "expr-voice-3-f": 0.8, "expr-voice-3-m": 0.8,
    "expr-voice-4-f": 0.8, "expr-voice-4-m": 0.9,
    "expr-voice-5-f": 0.8, "expr-voice-5-m": 0.8,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="Kiki")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="Noise seed (deterministic generator noise).")
    ap.add_argument("--out", default="tmp/torch.wav")
    args = ap.parse_args()

    voice_id = VOICE_ALIASES.get(args.voice, args.voice)
    effective_speed = args.speed * SPEED_PRIORS.get(voice_id, 1.0)

    print(f"loading weights: {WEIGHTS}")
    bag = WeightBag.load(WEIGHTS)
    model = KittenTTS(bag).eval()

    voices = np.load(str(VOICES_NPZ))
    voice_table = voices[voice_id].astype(np.float32)  # (400, 256)

    vocab = build_vocab()
    ids = phonemize(args.text, vocab)
    print(f"phonemes ({len(ids)}): {ids[:20]}...")

    # Style row indexed by phoneme count (mirrors runtime).
    ref_id = min(len(ids), voice_table.shape[0] - 1)
    style = voice_table[ref_id].reshape(1, 256)

    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    style_t = torch.from_numpy(style)

    with torch.no_grad():
        wav, durs = model(ids_t, style_t,
                          speed=effective_speed, noise_seed=args.seed)

    samples = wav.detach().cpu().numpy().astype(np.float32)
    # Float -> int16 wav
    samples = np.clip(samples, -1.0, 1.0)
    i16 = (samples * 32767.0).astype(np.int16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_path, 24000, i16)
    dur = len(samples) / 24000.0
    print(f"wrote {out_path}  len={len(samples)}  dur={dur:.3f}s  frames={int(durs.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
