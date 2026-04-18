#!/usr/bin/env python3
"""Check the tail of a wav for a transient / click that might sound as a crack.

We look at the last N ms of signal in three ways:
  1. Max absolute sample in the final window.
  2. Abrupt sample-to-sample jump (|x[i] - x[i-1]|) profile.
  3. Running RMS — is the signal fading out or does it stop abruptly?
"""
from __future__ import annotations
import sys, wave
import numpy as np
from pathlib import Path


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        bp = w.getsampwidth()
        raw = w.readframes(n)
    if bp == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    else:
        x = np.frombuffer(raw, dtype=np.float32)
    return x, sr


def tail_report(path: str) -> None:
    x, sr = read_wav(path)
    print(f"\n=== {path}  len={len(x)}  sr={sr}  dur={len(x)/sr:.2f}s ===")
    # Last 120 ms
    tail = x[-int(sr * 0.12):]
    print(f"last 120ms: max|x| = {np.max(np.abs(tail)):.3f}, rms = {np.sqrt((tail**2).mean()):.3f}")
    # Last 10 ms
    last10 = x[-int(sr * 0.01):]
    print(f"last 10ms:  max|x| = {np.max(np.abs(last10)):.3f}, rms = {np.sqrt((last10**2).mean()):.3f}")
    # Sample-to-sample deltas in final 20ms
    last20 = x[-int(sr * 0.02):]
    d = np.abs(np.diff(last20))
    print(f"last 20ms:  max |Δ| = {d.max():.3f}  "
          f"position of max = {len(x) - len(last20) + int(np.argmax(d))}")
    # Trailing samples
    print(f"final 16 samples: {[f'{v:.3f}' for v in x[-16:]]}")


def main():
    for p in sys.argv[1:]:
        if Path(p).exists():
            tail_report(p)


if __name__ == "__main__":
    main()
