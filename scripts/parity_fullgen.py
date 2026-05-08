#!/usr/bin/env python3
"""Parity test: kittens-tts (ggml) full generator (with noise) vs torch.

Replicates the GeneratorStage's full forward path: compute_noise_contribs +
GeneratorPipeline. Dumps the audio to a WAV file so it can be listened to.
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
from torch_kitten import (  # noqa: E402
    WeightBag, GeneratorPipeline, AdaINResBlockHiFiGAN, compute_noise_contribs,
)

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
GGUF         = REPO_ROOT / "tmp/kitten_full.gguf"
TMP          = REPO_ROOT / "tmp"


def torch_reference(dec_out: np.ndarray, f0_proj: np.ndarray,
                    style128: np.ndarray) -> np.ndarray:
    F = dec_out.shape[1] // 2
    w = WeightBag.load(str(SAFE_TENSORS))
    nr0_mod = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.0")
    nr1_mod = AdaINResBlockHiFiGAN(w, "decoder.generator.noise_res.1")
    gen = GeneratorPipeline(w).eval()

    dec_t = torch.from_numpy(dec_out).reshape(1, 256, 2 * F)
    f0_t  = torch.from_numpy(f0_proj).reshape(1, 1, 2 * F)
    s_t   = torch.from_numpy(style128).reshape(1, 128)
    with torch.no_grad():
        nr0, nr1 = compute_noise_contribs(f0_t, F, s_t, w, nr0_mod, nr1_mod, seed=None)
        audio = gen(dec_t, s_t, nr0, nr1)
    return audio.numpy()


def write_wav(path: Path, audio: np.ndarray, sr: int = 24000) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-F", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="cpu")
    args = ap.parse_args()

    F = args.F
    rng = np.random.default_rng(args.seed)
    dec_out = rng.standard_normal((256, 2 * F)).astype(np.float32) * 0.5
    # Synthesize a plausible f0 contour so voiced-mask kicks in: most positive,
    # with a few unvoiced gaps. Range ~80-200 Hz.
    f0 = (rng.standard_normal(2 * F).astype(np.float32) * 30 + 130)
    f0[:2] = 0.0   # leading silence
    f0[-2:] = 0.0  # trailing silence
    f0 = f0.reshape(1, 2 * F)
    style = rng.standard_normal(128).astype(np.float32) * 0.1
    print(f"F={F}, expected audio length = {600*F}")

    print("[torch] running full GeneratorStage path …")
    ref = torch_reference(dec_out, f0, style)
    print(f"[torch] audio shape={ref.shape}  audio[:6]={ref[:6]}")

    in_path = TMP / "fg_in.bin"
    out_path = TMP / "fg_out.bin"
    with in_path.open("wb") as fp:
        fp.write(struct.pack("<i", F))
        fp.write(dec_out.T.astype(np.float32).tobytes())   # NLC: transpose to data[t*C+c]
        fp.write(f0.T.astype(np.float32).tobytes())        # NLC
        fp.write(style.tobytes())

    bin_path = TMP / f"kittens-tts-{args.backend}"
    cmd = [str(bin_path), "--gguf", str(GGUF), "--mode", "fullgen",
           "--input", str(in_path), "--output", str(out_path),
           "--backend", args.backend]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout, end="")
    if r.stderr: print(r.stderr, end="", file=sys.stderr)
    if r.returncode != 0: raise SystemExit(f"exited {r.returncode}")

    with out_path.open("rb") as fp:
        T_g, = struct.unpack("<i", fp.read(4))
        got = np.frombuffer(fp.read(), dtype=np.float32)
    print(f"[ggml]  audio T={T_g}  audio[:6]={got[:6]}")

    if T_g != ref.size:
        print(f"LENGTH MISMATCH: got {T_g} expected {ref.size}")
        sys.exit(1)

    a, b = ref.flatten(), got.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    max_err = float(np.max(np.abs(a - b)))
    mean_err = float(np.mean(np.abs(a - b)))
    print(f"[fullgen] cos={cos:.6f}  max_err={max_err:.3e}  mean_err={mean_err:.3e}")

    # Always dump WAVs so we can listen, regardless of pass/fail.
    write_wav(TMP / "fullgen_torch.wav", ref)
    write_wav(TMP / "fullgen_ggml.wav",  got)
    print(f"wrote tmp/fullgen_torch.wav  tmp/fullgen_ggml.wav (sr=24000)")

    if cos < 0.99:
        print("FAIL"); sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
