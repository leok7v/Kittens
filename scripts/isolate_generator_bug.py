#!/usr/bin/env python3
"""Feed one fixed set of (prosody_lr, text_lr, style) tuples into BOTH
torch GeneratorStage and CoreML GeneratorStage, save both wavs, compare.

Since TextStage already matches to 1e-6 and iSTFT head is bit-exact, the
~720 Hz CoreML artifact must originate inside the generator — upsamplers,
AdaIN resblocks, or the noise path. Running both with the same upstream
inputs (and torch with a fixed noise seed) isolates that.

Writes tmp/torch_gen.wav and tmp/coreml_gen.wav for spectral A/B.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_kitten import TextStage, GeneratorStage, WeightBag  # noqa: E402

SR = 24000
L_BUCKET = 128
N_BUCKET = 256


def build_alignment(durs: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    n_frames = int(durs.sum())
    align = np.zeros((1, L, n_frames), dtype=np.float32)
    j = 0
    for i, d in enumerate(durs.tolist()):
        align[0, i, j:j + int(d)] = 1.0
        j += int(d)
    return align, n_frames


def pad_to(x: np.ndarray, axis: int, n: int) -> np.ndarray:
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, max(0, n - x.shape[axis]))
    y = np.pad(x, pad)
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(0, n)
    return y[tuple(slicer)]


def save_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    i16 = (x * 32767.0).astype(np.int16)
    wavfile.write(path, SR, i16)
    print(f"  wrote {path}  len={len(x)}  dur={len(x)/SR:.3f}s")


def main():
    import coremltools as ct

    weights = Path("Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors")
    voices = Path("scripts/models/voices.npz")
    text_pkg = f"scripts/models/kitten_text_L{L_BUCKET}.mlpackage"
    gen_pkg = f"scripts/models/kitten_generator_N{N_BUCKET}.mlpackage"

    print(f"loading weights + CoreML packages ...")
    bag = WeightBag.load(weights)
    torch_text = TextStage(bag).eval()
    torch_gen = GeneratorStage(bag).eval()
    cml_text = ct.models.MLModel(text_pkg)
    cml_gen = ct.models.MLModel(gen_pkg)

    vdata = np.load(str(voices))
    style_np = vdata["expr-voice-5-f"][L_BUCKET:L_BUCKET + 1].astype(np.float32)
    style_t = torch.from_numpy(style_np)

    # Use deterministic synthetic ids so both runs see identical input.
    rng = np.random.default_rng(0)
    ids_np = rng.integers(1, 170, size=(1, L_BUCKET), dtype=np.int32)
    ids_t = torch.from_numpy(ids_np.astype(np.int64))
    mask_np = np.ones((1, L_BUCKET), dtype=np.float32)
    mask_t = torch.from_numpy(mask_np)

    # Shared TextStage (torch — already proven bit-identical to CoreML).
    print("running torch TextStage ...")
    with torch.no_grad():
        p_t, tf_t, ds_t = torch_text(ids_t, style_t, mask_t)

    # Durations + alignment + padding.
    durs = np.maximum(1, np.round(ds_t.numpy().sum(axis=-1)[0]).astype(np.int32))
    align, n_frames = build_alignment(durs, L_BUCKET)
    print(f"  nFrames={n_frames}  sum(durs)={durs.sum()}")
    prosody_lr_np = (p_t.numpy() @ align).astype(np.float32)
    text_lr_np = (tf_t.numpy() @ align).astype(np.float32)
    prosody_lr_np = pad_to(prosody_lr_np, 2, N_BUCKET)
    text_lr_np = pad_to(text_lr_np, 2, N_BUCKET)
    prosody_lr_t = torch.from_numpy(prosody_lr_np)
    text_lr_t = torch.from_numpy(text_lr_np)

    print("running torch GeneratorStage (seed=0) ...")
    torch.manual_seed(0)
    with torch.no_grad():
        wav_t = torch_gen(prosody_lr_t, text_lr_t, style_t).numpy()
    save_wav(Path("tmp/torch_gen.wav"), wav_t)

    print("running CoreML GeneratorStage ...")
    gen_out = cml_gen.predict({
        "prosody_lr": prosody_lr_np,
        "text_lr": text_lr_np,
        "style": style_np,
    })
    wav_ml = next(iter(gen_out.values())).astype(np.float32).reshape(-1)
    save_wav(Path("tmp/coreml_gen.wav"), wav_ml)

    # Spectral analysis at key bins.
    n = min(len(wav_t), len(wav_ml))
    a = wav_t[:n].astype(np.float64)
    b = wav_ml[:n].astype(np.float64)
    err = b - a

    win = np.hanning(n)
    spec_a = np.abs(np.fft.rfft(a * win))
    spec_b = np.abs(np.fft.rfft(b * win))
    spec_e = np.abs(np.fft.rfft(err * win))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)

    cos = float((a * b).sum() /
                (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-30))
    print(f"\n[generator-stage] cos={cos:.4f}  rms_err={np.sqrt((err ** 2).mean()):.4f}")
    print(f"                   torch_rms={np.sqrt((a ** 2).mean()):.4f}  "
          f"coreml_rms={np.sqrt((b ** 2).mean()):.4f}")

    # Energy in ~700 Hz region to see if it's CoreML-amplified.
    bands = [(650, 700), (700, 750), (750, 800), (800, 900),
             (1400, 1600), (3000, 5000)]
    print(f"\n{'band_Hz':>12} {'torch_dB':>10} {'coreml_dB':>11} {'err_dB':>10}")
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        ta = 10 * np.log10((spec_a[mask] ** 2).mean() + 1e-12)
        tb = 10 * np.log10((spec_b[mask] ** 2).mean() + 1e-12)
        te = 10 * np.log10((spec_e[mask] ** 2).mean() + 1e-12)
        print(f"  {lo:>4d}-{hi:<4d} {ta:>10.1f} {tb:>11.1f} {te:>10.1f}")


if __name__ == "__main__":
    main()
