#!/usr/bin/env python3
"""
safetensors → GGUF converter for the KittenTTS Albert/BERT encoder.

Milestone 1 scope: only the BERT TextStage encoder. No heads, no generator.
Loads the safetensors with the same conventions as scripts/torch_kitten.py
(WeightBag.dequant / .bias / .f32) so the resulting GGUF is a faithful copy
of what BertStack consumes.

Usage:
    python scripts/convert_bert_to_gguf.py \
        --in  Sources/KittenApp/Resources/nano/kitten_tts_nano_v0_8.safetensors \
        --out scripts/models/kitten_bert.gguf

Architecture metadata (also written into the GGUF):
    kittens-bert
        vocab_size        178
        max_position      512
        token_types       2
        embedding_dim     128
        hidden_size       768
        num_layers        12   (shared Albert block, applied 12x)
        num_heads         12
        head_dim          64
        ffn_dim           2048
        layer_norm_eps    1e-12

Tensor naming (consumed by kittens-tts.c):
    embd.word.weight       (vocab,    embd_dim)   F32
    embd.pos.weight        (max_pos,  embd_dim)   F32
    embd.type.weight       (2,        embd_dim)   F32
    embd.ln.weight         (embd_dim,)            F32
    embd.ln.bias           (embd_dim,)            F32
    embd_to_hidden.weight  (hidden,   embd_dim)   F32   numpy = (out, in)
    embd_to_hidden.bias    (hidden,)              F32
    layer.attn_q.weight    (hidden,   hidden)     F16   numpy = (out, in)
    layer.attn_q.bias      (hidden,)              F32
    layer.attn_k.weight    (hidden,   hidden)     F16   numpy = (out, in)
    layer.attn_k.bias      (hidden,)              F32
    layer.attn_v.weight    (hidden,   hidden)     F16   numpy = (out, in)
    layer.attn_v.bias      (hidden,)              F32
    layer.attn_out.weight  (hidden,   hidden)     F32   numpy = (out, in)
    layer.attn_out.bias    (hidden,)              F32
    layer.attn_ln.weight   (hidden,)              F32
    layer.attn_ln.bias     (hidden,)              F32
    layer.ffn.weight       (ffn,      hidden)     F16   numpy = (out, in)
    layer.ffn.bias         (ffn,)                 F32
    layer.ffn_out.weight   (hidden,   ffn)        F16   numpy = (out, in)
    layer.ffn_out.bias     (hidden,)              F32
    layer.full_ln.weight   (hidden,)              F32
    layer.full_ln.bias     (hidden,)              F32

ONNX MatMul weights are stored as numpy (in, out). ggml reverses dim order
(numpy last axis = ggml ne[0]), and ggml_mul_mat(W, x) expects W's ne[0] to
be the contracting (input) dim. So we save matmul weights TRANSPOSED in
numpy ((out, in)) so ggml interprets them as ne=(in, out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "vendors" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


ARCH = "kittens-bert"


def _t(f, key: str) -> np.ndarray:
    """Load a tensor as fp32 numpy. Raises if missing."""
    if key not in f.keys():
        raise KeyError(f"missing tensor: {key}")
    t = f.get_tensor(key)
    return t.float().numpy() if t.dtype.is_floating_point else t.numpy()


def _t_native(f, key: str) -> np.ndarray:
    """Load a tensor in its native dtype. Used for fp16 matmul weights so we
    don't round-trip through fp32 just to cast back."""
    if key not in f.keys():
        raise KeyError(f"missing tensor: {key}")
    t = f.get_tensor(key)
    return t.numpy()


def convert(src: Path, dst: Path) -> None:
    print(f"reading {src}")
    with safe_open(str(src), framework="pt") as f:
        # --- embeddings (all fp32 in source) ---
        we = _t(f, "kmodel.bert.embeddings.word_embeddings.weight")
        pe = _t(f, "kmodel.bert.embeddings.position_embeddings.weight")
        te = _t(f, "kmodel.bert.embeddings.token_type_embeddings.weight")
        ln_w = _t(f, "kmodel.bert.embeddings.LayerNorm.weight")
        ln_b = _t(f, "kmodel.bert.embeddings.LayerNorm.bias")

        # --- 128 -> 768 mapping (fp32) ---
        m_in = _t(f, "onnx::MatMul_5661")  # (128, 768)
        m_in_b = _t(f, "kmodel.bert.encoder.embedding_hidden_mapping_in.bias")

        # --- shared Albert block ---
        base = "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0"
        # Matmul weights kept native (fp16 where source is fp16 — saves 50% on
        # disk for the heavy tensors and matches how ggml stores them).
        qW = _t_native(f, "onnx::MatMul_5662")          # fp16 (768, 768)
        kW = _t_native(f, "onnx::MatMul_5665")          # fp16 (768, 768)
        vW = _t_native(f, "onnx::MatMul_5668")          # fp16 (768, 768)
        oW = _t_native(f, "onnx::MatMul_5672")          # fp32 (768, 768) — keep fp32
        ffnW = _t_native(f, "onnx::MatMul_5673")        # fp16 (768, 2048)
        ffnOutW = _t_native(f, "onnx::MatMul_5674")     # fp16 (2048, 768)

        # Biases (cast to fp32 — small, simplifies ggml_add).
        qB = _t(f, f"{base}.attention.query.bias")
        kB = _t(f, f"{base}.attention.key.bias")
        vB = _t(f, f"{base}.attention.value.bias")
        oB = _t(f, f"{base}.attention.dense.bias")
        ffnB = _t(f, f"{base}.ffn.bias")
        ffnOutB = _t(f, f"{base}.ffn_output.bias")

        # LayerNorms (fp32).
        attn_ln_w = _t(f, f"{base}.attention.LayerNorm.weight")
        attn_ln_b = _t(f, f"{base}.attention.LayerNorm.bias")
        full_ln_w = _t(f, f"{base}.full_layer_layer_norm.weight")
        full_ln_b = _t(f, f"{base}.full_layer_layer_norm.bias")

    # --- sanity checks ---
    assert we.shape == (178, 128), we.shape
    assert pe.shape == (512, 128), pe.shape
    assert te.shape == (2, 128), te.shape
    assert m_in.shape == (128, 768), m_in.shape
    assert qW.shape == (768, 768) and kW.shape == (768, 768) and vW.shape == (768, 768)
    assert oW.shape == (768, 768)
    assert ffnW.shape == (768, 2048) and ffnOutW.shape == (2048, 768)

    print(f"writing {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(dst), ARCH)

    # --- arch metadata ---
    w.add_uint32("kittens-bert.vocab_size", 178)
    w.add_uint32("kittens-bert.max_position", 512)
    w.add_uint32("kittens-bert.token_types", 2)
    w.add_uint32("kittens-bert.embedding_dim", 128)
    w.add_uint32("kittens-bert.hidden_size", 768)
    w.add_uint32("kittens-bert.num_layers", 12)
    w.add_uint32("kittens-bert.num_heads", 12)
    w.add_uint32("kittens-bert.head_dim", 64)
    w.add_uint32("kittens-bert.ffn_dim", 2048)
    w.add_float32("kittens-bert.layer_norm_eps", 1e-12)

    # --- tensors ---
    def add(name: str, arr: np.ndarray, want_fp16: bool = False) -> None:
        if want_fp16 and arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        if not want_fp16 and arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        if arr.dtype not in (np.float16, np.float32):
            arr = arr.astype(np.float32)
        w.add_tensor(name, np.ascontiguousarray(arr))

    # embeddings
    add("embd.word.weight", we)
    add("embd.pos.weight", pe)
    add("embd.type.weight", te)
    add("embd.ln.weight", ln_w)
    add("embd.ln.bias", ln_b)

    # 128 -> 768 mapping (ONNX in,out → save as numpy out,in for ggml)
    add("embd_to_hidden.weight", m_in.T)
    add("embd_to_hidden.bias", m_in_b)

    # shared Albert block — matmul weights transposed to (out, in) in numpy
    # so ggml interprets them as (in, out) with the input dim as ne[0].
    add("layer.attn_q.weight", qW.T, want_fp16=True)
    add("layer.attn_q.bias", qB)
    add("layer.attn_k.weight", kW.T, want_fp16=True)
    add("layer.attn_k.bias", kB)
    add("layer.attn_v.weight", vW.T, want_fp16=True)
    add("layer.attn_v.bias", vB)
    add("layer.attn_out.weight", oW.T)  # fp32 source
    add("layer.attn_out.bias", oB)
    add("layer.attn_ln.weight", attn_ln_w)
    add("layer.attn_ln.bias", attn_ln_b)

    add("layer.ffn.weight", ffnW.T, want_fp16=True)
    add("layer.ffn.bias", ffnB)
    add("layer.ffn_out.weight", ffnOutW.T, want_fp16=True)
    add("layer.ffn_out.bias", ffnOutB)
    add("layer.full_ln.weight", full_ln_w)
    add("layer.full_ln.bias", full_ln_b)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"done. {dst} = {size_mb:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    args = ap.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
