#!/usr/bin/env python3
"""Dump a fixture for testing the bidirectional LSTM primitive in kittens-tts.

Exports:
    tmp/lstm_fixture.gguf  - weights for one LSTM (duration_lstm by default)
    tmp/lstm_fixture_in.bin - random input (T, in_size) fp32
    tmp/lstm_fixture_ref.bin - torch nn.LSTM output (T, 2H) fp32

GGUF layout (architecture name "kittens-lstm-test"):
    metadata: in_size, hidden, T
    tensors:
        lstm.fwd.W   (4H, in)   F32  - weight_ih_l0   (PyTorch ifgo order)
        lstm.fwd.R   (4H, H)    F32  - weight_hh_l0
        lstm.fwd.b   (4H,)      F32  - bias_ih + bias_hh  (combined)
        lstm.bwd.W   (4H, in)   F32  - weight_ih_l0_reverse
        lstm.bwd.R   (4H, H)    F32
        lstm.bwd.b   (4H,)      F32

Tensors are saved as numpy (out, in) so ggml interprets them as (in, out) —
same convention as convert_bert_to_gguf.py.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "vendors" / "llama.cpp" / "gguf-py"))
from torch_kitten import WeightBag, load_onnx_bidir_lstm  # noqa: E402
import gguf  # noqa: E402

SAFE_TENSORS = REPO_ROOT / "Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors"
ARCH = "kittens-lstm-test"

# Default: the duration LSTM (in=256, H=64).
LSTMS = {
    "duration": ("onnx::LSTM_5971", "onnx::LSTM_5972", "onnx::LSTM_5970"),
    "shared":   ("onnx::LSTM_6020", "onnx::LSTM_6021", "onnx::LSTM_6019"),
    "ptext0":   ("onnx::LSTM_5872", "onnx::LSTM_5873", "onnx::LSTM_5871"),
    "ptext2":   ("onnx::LSTM_5922", "onnx::LSTM_5923", "onnx::LSTM_5921"),
    "atext":    ("onnx::LSTM_5652", "onnx::LSTM_5653", "onnx::LSTM_5651"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="duration", choices=list(LSTMS))
    ap.add_argument("-T", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-gguf", default="tmp/lstm_fixture.gguf")
    ap.add_argument("--out-in",   default="tmp/lstm_fixture_in.bin")
    ap.add_argument("--out-ref",  default="tmp/lstm_fixture_ref.bin")
    args = ap.parse_args()

    w = WeightBag.load(str(SAFE_TENSORS))
    keys = LSTMS[args.name]
    bidir = load_onnx_bidir_lstm(w, *keys)
    lstm = bidir.lstm.eval()
    H = bidir.hidden_size
    in_size = lstm.weight_ih_l0.shape[1]
    T = args.T
    print(f"LSTM '{args.name}': in_size={in_size} H={H} T={T}")

    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal((T, 1, in_size)).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    with torch.no_grad():
        y_t, _ = lstm(x_t)        # (T, 1, 2H)
    y_np = y_t.detach().numpy().reshape(T, 2 * H)

    # ---- write input/output ----
    Path(args.out_in).parent.mkdir(parents=True, exist_ok=True)
    x_np.reshape(T, in_size).tofile(args.out_in)
    y_np.tofile(args.out_ref)
    print(f"wrote {args.out_in}  ({x_np.size} f32)")
    print(f"wrote {args.out_ref} ({y_np.size} f32)")

    # ---- pull weights & write GGUF ----
    def numpy_t(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    fwd_W = numpy_t(lstm.weight_ih_l0)             # (4H, in)
    fwd_R = numpy_t(lstm.weight_hh_l0)             # (4H, H)
    fwd_b = numpy_t(lstm.bias_ih_l0 + lstm.bias_hh_l0)  # (4H,)
    bwd_W = numpy_t(lstm.weight_ih_l0_reverse)
    bwd_R = numpy_t(lstm.weight_hh_l0_reverse)
    bwd_b = numpy_t(lstm.bias_ih_l0_reverse + lstm.bias_hh_l0_reverse)

    assert fwd_W.shape == (4 * H, in_size), fwd_W.shape
    assert fwd_R.shape == (4 * H, H), fwd_R.shape

    gw = gguf.GGUFWriter(args.out_gguf, ARCH)
    gw.add_uint32("kittens-lstm-test.in_size", in_size)
    gw.add_uint32("kittens-lstm-test.hidden",  H)
    gw.add_uint32("kittens-lstm-test.T",       T)

    # numpy already (4H, in) — ggml interprets as (in, 4H) which is what
    # ggml_mul_mat(W, x) wants.
    gw.add_tensor("lstm.fwd.W", np.ascontiguousarray(fwd_W))
    gw.add_tensor("lstm.fwd.R", np.ascontiguousarray(fwd_R))
    gw.add_tensor("lstm.fwd.b", np.ascontiguousarray(fwd_b))
    gw.add_tensor("lstm.bwd.W", np.ascontiguousarray(bwd_W))
    gw.add_tensor("lstm.bwd.R", np.ascontiguousarray(bwd_R))
    gw.add_tensor("lstm.bwd.b", np.ascontiguousarray(bwd_b))

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"wrote {args.out_gguf}")


if __name__ == "__main__":
    main()
