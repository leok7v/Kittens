#!/usr/bin/env python3
"""Compare two KittenTTS backend outputs (e.g. output_mlx.wav vs output_coreml.wav).

Raw waveform comparison is misleading because both backends inject independent
sine-gen noise. Instead we compare log-mel-spectrograms — that captures
whether the acoustic content (phoneme energies, pitch contour) agrees, which
is what the model is actually producing.

Usage:
    python compare_backends.py output_mlx.wav output_coreml.wav
    python compare_backends.py --plot output_mlx.wav output_coreml.wav
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        bytes_per = wf.getsampwidth()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    if bytes_per == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    elif bytes_per == 4:
        samples = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unsupported bit depth: {bytes_per * 8}")
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    return samples, sr


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _mel_filter_bank(n_mels: int, n_fft: int, sr: int,
                     fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m_min = hz_to_mel(fmin); m_max = hz_to_mel(fmax)
    m = np.linspace(m_min, m_max, n_mels + 2)
    hz = mel_to_hz(m)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        for k in range(l, c):
            fb[i, k] = (k - l) / (c - l)
        for k in range(c, r):
            fb[i, k] = (r - k) / (r - c)
    return fb


def log_mel_spectrogram(x: np.ndarray, sr: int, n_fft: int = 1024,
                         hop: int = 256, n_mels: int = 80) -> np.ndarray:
    # Pad so framing covers the signal.
    pad = n_fft // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    window = _hann(n_fft).astype(np.float32)
    frames = np.empty((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        frames[i] = x[i * hop : i * hop + n_fft] * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec).astype(np.float32)
    fb = _mel_filter_bank(n_mels, n_fft, sr)
    mel = mag @ fb.T
    log_mel = np.log(mel + 1e-6)
    return log_mel  # (T, n_mels)


def summary(label: str, x: np.ndarray, sr: int) -> None:
    dur = len(x) / sr
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    print(f"  {label:8s} {dur:5.2f}s  {len(x):6d} samples  rms {rms:.4f}  peak {peak:.4f}")


def _resample_frames(x: np.ndarray, n: int) -> np.ndarray:
    """Linear-interpolate a (T, D) mel spectrogram to n frames along time."""
    T, D = x.shape
    if T == n:
        return x
    src = np.linspace(0.0, T - 1, num=n)
    i0 = np.floor(src).astype(int)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (src - i0)[:, None]
    return (1.0 - w) * x[i0] + w * x[i1]


def compare(a: np.ndarray, b: np.ndarray, sr: int) -> None:
    ma = log_mel_spectrogram(a, sr)
    mb = log_mel_spectrogram(b, sr)
    # The two backends emit different-length audio because noise and rounding
    # shift nFrames slightly. Resample B's spectrogram onto A's frame axis so
    # a per-frame comparison is meaningful.
    mb_aligned = _resample_frames(mb, ma.shape[0])
    diff = ma - mb_aligned
    # MCD: Mel-Cepstral Distortion uses DCT coefficients; we approximate with
    # log-mel RMS (×10/ln10 for dB), which tracks MCD closely in practice.
    mcd = float(np.sqrt(np.mean(diff ** 2)) * (10.0 / np.log(10.0)))
    num = (ma * mb_aligned).sum(axis=1)
    den = np.sqrt((ma ** 2).sum(axis=1) * (mb_aligned ** 2).sum(axis=1)) + 1e-30
    fcos = float((num / den).mean())
    gcos = float(((ma * mb_aligned).sum())
                 / (np.sqrt((ma ** 2).sum() * (mb_aligned ** 2).sum()) + 1e-30))
    print(f"\nLog-mel spectrogram diff (aligned, {ma.shape[0]} frames):")
    print(f"  per-frame cosine (mean): {fcos:.4f}")
    print(f"  global cosine:            {gcos:.4f}")
    print(f"  pseudo-MCD:               {mcd:.2f} dB  (< 6 is usually inaudible, > 10 is bad)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--plot", action="store_true",
                    help="save mel spectrogram PNGs alongside the inputs")
    args = ap.parse_args()

    xa, sra = read_wav(args.a)
    xb, srb = read_wav(args.b)
    if sra != srb:
        print(f"warning: sample-rate mismatch {sra} vs {srb}", file=sys.stderr)
    sr = min(sra, srb)

    print("File summary:")
    summary(args.a.name, xa, sra)
    summary(args.b.name, xb, srb)

    compare(xa, xb, sr)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not installed; skipping --plot", file=sys.stderr)
            return 0
        for path, x, sr_ in ((args.a, xa, sra), (args.b, xb, srb)):
            m = log_mel_spectrogram(x, sr_)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(m.T, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(path.name)
            out = path.with_suffix(".mel.png")
            fig.tight_layout()
            fig.savefig(str(out), dpi=120)
            plt.close(fig)
            print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
