#!/usr/bin/env python3
"""Produce REAL (prosody_lr, text_lr, style) calibration tuples per N-bucket.

Background: int8wa quantization previously used Gaussian-noise tuples to
calibrate the generator. That destroyed the length regulator / F0 / noise
paths — audio became "cave echo" or "low-volume whisper". The activation
quantizer picks per-tensor scales from the observed max, and Gaussians
don't reproduce the temporal structure (repeated-frame prosody, phoneme
plateaus, zero-tail silence) of real length-regulated features.

This script runs the FP32 PyTorch TextStage (torch_kitten.py) on a small
corpus of real English sentences, length-regulates in Python exactly like
Swift does at runtime, and dumps the padded (prosody_lr, text_lr, style)
tuples to disk for calibrate_and_quantize.py to consume.

Usage:
    python scripts/real_calibration.py                # default corpus
    python scripts/real_calibration.py --out scripts/models/calibration_tuples
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from torch_kitten import TextStage, WeightBag

WEIGHTS = Path("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
VOICES_NPZ = Path("scripts/models/voices.npz")

TEXT_BUCKETS = [16, 32, 64, 128, 400]
GEN_BUCKETS = [128, 256, 512, 1024]

# Mirror Sources/KittenApp/TTS.swift `Phonemizer.symbols`. Keep in sync if
# the vocab changes upstream — this is a runtime-determined mapping, not a
# saved asset.
_PUNCT = ";:,.!?\u00a1\u00bf\u2014\u2026\"\u00ab\u00bb\u201d\u201d "
_ASCII = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_IPA = ("\u0251\u0250\u0252\u00e6\u0253\u02b9\u03b2\u0254\u0255\u00e7\u0257"
        "\u0256\u00f0\u02a4\u0259\u0258\u025a\u025b\u025c\u025d\u025e\u025f"
        "\u0284\u0261\u0260\u0262\u029b\u0266\u0267\u0127\u0265\u029c\u0268"
        "\u026a\u029d\u026d\u026c\u026e\u027f\u0271\u026f\u0270\u014b\u0273"
        "\u0272\u0274\u00f8\u0275\u0278\u03b8\u0153\u0276\u0298\u0299\u027b"
        "\u027a\u027e\u027b\u0280\u0281\u027d\u0282\u0283\u0288\u02a7\u0289"
        "\u028a\u028b\u2c71\u028c\u0263\u0264\u028d\u03c7\u028e\u028f\u0291"
        "\u0290\u0292\u0294\u02a1\u0295\u02a2\u01c0\u01c1\u01c2\u01c3\u02c8"
        "\u02cc\u02d0\u02d1\u02bc\u02b4\u02b0\u02b1\u02b2\u02b7\u02e0\u02e4"
        "\u02de\u2193\u2191\u2192\u2197\u2198'\u0329'\u1d7b")


def build_vocab() -> dict[str, int]:
    symbols = ["$"]
    for ch in _PUNCT:
        symbols.append(ch)
    for ch in _ASCII:
        symbols.append(ch)
    for ch in _IPA:
        symbols.append(ch)
    # Dictionary with later-wins semantics (matches Swift's
    # `uniquingKeysWith: { _, new in new }`).
    d: dict[str, int] = {}
    for i, s in enumerate(symbols):
        d[s] = i
    return d


def espeak_ipa(text: str) -> str:
    """Call espeak-ng --ipa -q and flatten multi-line output."""
    out = subprocess.run(
        ["espeak-ng", "-v", "en-us", "--ipa", "-q", text],
        check=True, capture_output=True, text=True,
    ).stdout
    return " ".join(line.strip() for line in out.splitlines() if line.strip())


def phonemize(text: str, vocab: dict[str, int]) -> list[int]:
    """Match Swift's Phonemizer.phonemize: [0, ...ids..., 10, 0]."""
    ipa = espeak_ipa(text)
    tokens = [0]
    for ch in ipa:
        if ch in vocab:
            tokens.append(vocab[ch])
    tokens.append(10)
    tokens.append(0)
    return tokens


def durations_from_sig(dur_sig: torch.Tensor, speed: float, real_L: int) -> list[int]:
    """Replicates the Swift-side duration calculation:
       d[i] = max(1, round(sum_j dur_sig[0, i, j] / speed))."""
    durs: list[int] = []
    sig = dur_sig[0]  # (L, 50)
    for i in range(real_L):
        total = float(sig[i].sum().item()) / speed
        durs.append(max(1, int(round(total))))
    return durs


def length_regulate(prosody_ncl: torch.Tensor, text_feat_ncl: torch.Tensor,
                    durations: list[int], n_bucket: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Same expansion Swift does per frame: for each phoneme i, write its
    prosody/text vector `durations[i]` times into the next slot. Remaining
    frames up to n_bucket stay zero."""
    prosody = prosody_ncl[0].detach().cpu().numpy()   # (256, L)
    text = text_feat_ncl[0].detach().cpu().numpy()    # (128, L)
    L = prosody.shape[1]

    prosody_lr = np.zeros((1, 256, n_bucket), dtype=np.float32)
    text_lr = np.zeros((1, 128, n_bucket), dtype=np.float32)
    total = sum(durations)

    # Mirror Swift overflow-trim: shave from the tail if we exceed N.
    if total > n_bucket:
        overflow = total - n_bucket
        for i in range(len(durations) - 1, -1, -1):
            shrink = min(durations[i] - 1, overflow)
            durations[i] -= shrink
            overflow -= shrink
            if overflow <= 0:
                break

    f = 0
    for i, d in enumerate(durations):
        if i >= L:
            break
        for _ in range(d):
            if f >= n_bucket:
                break
            prosody_lr[0, :, f] = prosody[:, i]
            text_lr[0, :, f] = text[:, i]
            f += 1
        if f >= n_bucket:
            break
    return prosody_lr, text_lr


def pick_bucket(value: int, buckets: list[int]) -> int:
    for b in buckets:
        if b >= value:
            return b
    return buckets[-1]


# Short, medium, long sentences — mix of content types so prosody / text
# features span a range of distributions. 26 entries; multiple hit every
# N bucket.
CORPUS = [
    "Hello.",
    "Good morning.",
    "Thank you very much.",
    "Yes, of course.",
    "I am here.",
    "Please wait a moment.",
    "Kitten is listening.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "Peter Piper picked a peck of pickled peppers.",
    "The rain in Spain falls mainly on the plain.",
    "Speech synthesis has come a long way in recent years.",
    "Welcome to the neural network demonstration project.",
    "This sentence contains a few commas, some pauses, and a final period.",
    "Numbers like one two three four five can sometimes trip up a model.",
    "Kitten TTS is now streaming audio chunks for lower latency.",
    "On device machine learning preserves user privacy while keeping latency low.",
    "Apple Silicon has dedicated neural engines that accelerate matrix math.",
    "Quantization reduces weight storage but preserves most of the accuracy.",
    "Once upon a time, in a land far far away, there lived a small cat.",
    "The tiny kitten padded softly across the hardwood floor, tail held high.",
    "We are building a small, fast, and friendly text to speech system for everyone.",
    "Careful calibration of activation scales is essential for low bit precision.",
    "Speech models trained on large and diverse corpora generalize across voices and styles.",
    "In the beginning there was silence, and then, quite suddenly, there was speech.",
    # N=512 bucket targets: ~130 to 170 phonemes → roughly 280 to 500 frames.
    "The kitten curled up on the warm windowsill and watched the birds outside.",
    "Soft rain tapped against the glass as the small cat dreamed of fish and sunshine.",
    "Neural networks learn patterns from data, generalizing across voices and contexts.",
    "A good text to speech engine captures intonation, rhythm, emphasis, and natural pacing.",
    "Streaming audio chunks lets the user hear speech while later chunks are still being generated.",
    "Quantization trades a small amount of model accuracy for much smaller memory footprint.",
    "On device inference keeps user data private and eliminates any dependence on a remote server.",
    "The generator takes prosody and text features and emits a mono waveform at twenty four kilohertz.",
    # Medium-long passages aimed at N=512 bucket (~280..512 frames).
    "Running a neural speech model on a small device requires careful engineering to keep memory "
    "and latency low while still producing high quality natural sounding audio.",
    "Before the text to speech model can synthesize a sentence, the phonemizer must first map "
    "every word to its pronunciation using either a rule based engine or a trained grapheme to "
    "phoneme network.",
    "The generator stage of the synthesizer takes a length regulated prosody tensor and a length "
    "regulated text features tensor along with a style embedding, and uses them to produce raw "
    "mono audio at twenty four kilohertz.",
    "When quantizing a neural network down to eight bit integers, it is essential to run a "
    "calibration pass on a representative set of real inputs so that the activation scales are "
    "picked accurately and numerical accuracy is preserved throughout the network.",
    # Longer passages to populate N=512 / N=1024 buckets.
    "The neural speech synthesizer takes a string of text as input, converts it into a sequence "
    "of phonemes using a grapheme to phoneme model, and then feeds that sequence through an "
    "acoustic model to produce a waveform that sounds like natural human speech.",
    "When a user types a sentence into the application, the runtime first breaks the text into "
    "chunks at sentence boundaries, phonemizes each chunk, picks a length bucket large enough to "
    "contain the phoneme sequence, and runs the text stage model to produce prosody and duration "
    "predictions that feed into the generator stage.",
    "The quick brown fox jumps over the lazy dog, and then the dog gets up, stretches, yawns, "
    "walks around the yard, chases a butterfly, comes back to the porch, and settles down for "
    "another nap in the warm afternoon sunlight, while the fox disappears into the tall grass "
    "at the edge of the meadow, never to be seen again that day, but surely returning at dusk.",
    "Speech is one of the most fundamental ways humans communicate with each other, conveying "
    "not only the literal meaning of words but also emotion, intent, urgency, humor, and "
    "personality through prosody, pitch, and timing; a good text to speech system must capture "
    "all of these subtle cues in addition to getting the words themselves right, which is why "
    "researchers have spent decades developing better acoustic models, better voice embeddings, "
    "and better neural architectures to faithfully reproduce the nuance of human voice.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/models/calibration_tuples")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--voice", default="expr-voice-5-m")
    args = ap.parse_args()

    out_root = Path(args.out)
    for N in GEN_BUCKETS:
        (out_root / f"N{N}").mkdir(parents=True, exist_ok=True)
    for L in TEXT_BUCKETS:
        (out_root / f"L{L}").mkdir(parents=True, exist_ok=True)

    print(f"loading weights: {WEIGHTS}")
    bag = WeightBag.load(WEIGHTS)
    text_stage = TextStage(bag).eval()

    voices = np.load(str(VOICES_NPZ))
    if args.voice not in voices.files:
        raise SystemExit(f"voice '{args.voice}' not found; try: {voices.files}")
    voice_table = voices[args.voice].astype(np.float32)  # (400, 256)

    vocab = build_vocab()

    per_bucket: dict[int, int] = {n: 0 for n in GEN_BUCKETS}
    per_L: dict[int, int] = {l: 0 for l in TEXT_BUCKETS}

    with torch.no_grad():
        for idx, sentence in enumerate(CORPUS):
            ids = phonemize(sentence, vocab)
            real_L = len(ids)
            L = pick_bucket(real_L, TEXT_BUCKETS)
            if real_L > L:
                real_L = L
                ids = ids[:L]

            # Style row indexed by phoneme count (mirrors runtime).
            ref_id = min(len(ids), voice_table.shape[0] - 1)
            style = voice_table[ref_id].reshape(1, 256)

            ids_t = torch.zeros(1, L, dtype=torch.long)
            ids_t[0, :real_L] = torch.tensor(ids, dtype=torch.long)
            mask_t = torch.zeros(1, L, dtype=torch.float32)
            mask_t[0, :real_L] = 1.0
            style_t = torch.from_numpy(style)

            prosody_ncl, text_feat_ncl, dur_sig = text_stage(
                ids_t, style_t, mask_t)

            # Save text-stage inputs bucketed by L (for int8wa text calibration).
            text_slot = per_L[L]
            per_L[L] += 1
            text_npz = out_root / f"L{L}" / f"sample_{text_slot:03d}.npz"
            np.savez(text_npz,
                     input_ids=ids_t.detach().cpu().numpy().astype(np.int32),
                     attention_mask=mask_t.detach().cpu().numpy().astype(np.float32),
                     style=style.astype(np.float32))

            durs = durations_from_sig(dur_sig, args.speed, real_L)
            total_frames = sum(durs)
            N = pick_bucket(total_frames, GEN_BUCKETS)
            prosody_lr, text_lr = length_regulate(
                prosody_ncl, text_feat_ncl, list(durs), N)

            slot = per_bucket[N]
            per_bucket[N] += 1
            npz_path = out_root / f"N{N}" / f"sample_{slot:03d}.npz"
            np.savez(npz_path, prosody_lr=prosody_lr, text_lr=text_lr,
                     style=style)
            print(f"  [{idx:02d}] L={real_L:3d}->{L:3d}  frames={total_frames:4d}"
                  f" -> N{N}  [{npz_path.name}]  \"{sentence[:60]}\"")

    print("\nper-bucket counts:")
    for N in GEN_BUCKETS:
        print(f"  N={N}: {per_bucket[N]} samples")
    for L in TEXT_BUCKETS:
        print(f"  L={L}: {per_L[L]} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
