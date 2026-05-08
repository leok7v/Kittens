#!/usr/bin/env python3
"""End-to-end parity: ggml chained pipeline vs torch_kitten.KittenTTS.

Both consume the same phoneme IDs + style; produce audio. Compare cos.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from real_calibration import build_vocab, phonemize  # noqa: E402
from torch_kitten import KittenTTS, WeightBag  # noqa: E402

GGUF = REPO_ROOT / "tmp/kitten_full.gguf"
TMP  = REPO_ROOT / "tmp"
SR = 24000


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def run_ggml_e2e(text: str, voice: str) -> np.ndarray:
    out_path = TMP / "tts_e2e.wav"
    r = subprocess.run(
        ["python3", str(REPO_ROOT / "scripts/tts_e2e.py"),
         "--text", text, "--voice", voice, "--out", str(out_path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, end=""); print(r.stderr, end="", file=sys.stderr)
        raise SystemExit(f"e2e exited {r.returncode}")
    with wave.open(str(out_path), "rb") as f:
        n = f.getnframes()
        raw = f.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


def run_torch_e2e(text: str, voice: str) -> np.ndarray:
    """Run torch_kitten.KittenTTS end-to-end on the same input."""
    vocab = build_vocab()
    ids = phonemize(text, vocab)
    L = len(ids)

    voices = np.load(REPO_ROOT / "scripts/models/voices.npz")
    vrows = voices[voice].astype(np.float32)
    ref = min(len(text), 399)
    style = vrows[ref]

    safetensors = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
    w = WeightBag.load(str(safetensors))
    model = KittenTTS(w).eval()
    ids_t   = torch.tensor(ids, dtype=torch.long).reshape(1, -1)
    style_t = torch.from_numpy(style).reshape(1, 256)
    with torch.no_grad():
        wav, durs = model(ids_t, style_t, speed=1.0, noise_seed=None)
    return wav.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello world.")
    ap.add_argument("--voice", default="expr-voice-5-m")
    args = ap.parse_args()

    print("[ggml] running e2e …")
    g = run_ggml_e2e(args.text, args.voice)
    write_wav(TMP / "parity_e2e_ggml.wav", g)
    print(f"  ggml audio: {len(g)} samples ({len(g)/SR:.2f} s)")

    print("[torch] running KittenTTS …")
    t = run_torch_e2e(args.text, args.voice)
    write_wav(TMP / "parity_e2e_torch.wav", t)
    print(f"  torch audio: {len(t)} samples ({len(t)/SR:.2f} s)")

    n = min(len(g), len(t))
    g, t = g[:n], t[:n]
    cos = float(np.dot(g, t) / (np.linalg.norm(g)*np.linalg.norm(t) + 1e-12))
    me = float(np.max(np.abs(g - t)))
    print(f"\ncos={cos:.6f}  max_err={me:.3e}")
    print(f"WAVs: tmp/parity_e2e_torch.wav  tmp/parity_e2e_ggml.wav")


if __name__ == "__main__":
    main()
