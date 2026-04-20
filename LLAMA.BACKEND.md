# llama.cpp / ggml portable TTS backend — plan

Goal: a **separate new repo** that runs KittenTTS on top of llama.cpp /
ggml, targeting non-Apple hardware (Linux, Android) as first-class, with
macOS / iOS coming along for free via ggml-metal. This document lays out
what from the current Kittens repo will carry over, what needs to be
built new, and a phased plan we can execute against.

Not starting work yet — waiting on sign-off after this plan is reviewed.

## 0. Scope recap

- **Target platforms (priority order)**: Linux (x86_64, arm64),
  Android (arm64), macOS, iOS. Windows comes for free if we avoid
  POSIX-only code paths.
- **Model**: KittenTTS-nano v0.8 (the one we're shipping here).
  No other architectures in scope.
- **Runtime**: ggml + llama.cpp's GGUF machinery. No Python at inference
  time. Metal on Apple, CUDA / Vulkan / CPU-SIMD elsewhere.
- **Repo shape**: new repo, e.g. `kittens-ggml` or similar. Apache/MIT
  license to match upstream KittenTTS and llama.cpp.

## 1. What from THIS repo carries over

### 1A. Drops in verbatim (no changes needed)

| Asset | Current location | Why it transfers |
|---|---|---|
| CEPhonemizer C++ engine | `Sources/KittenApp/CEPhonemizer/{phonemizer.h,phonemizer.cpp,ipa_table.h,rule_parser.h,include/}` | Already platform-neutral C++. Depends on `en_rules` and `en_list` files only. **Drop `swift_bridge.cpp`** — that's the only Apple-specific file. |
| en_rules + en_list | `Sources/KittenApp/Resources/nano/` | Plain text rule tables. Portable. |
| voices.safetensors | `Sources/KittenApp/Resources/nano/` | 8 voice embeddings × 400 × 256 float. Portable; we'll expose as a CLI flag `--voice`. |
| kitten_tts_nano_v0_8.safetensors | `Sources/KittenApp/Resources/nano/` | Model weights. Input to the new GGUF converter. |
| Sample prompts | `Sources/KittenApp/Resources/prompts/*.txt` | Test corpus. Copy as-is. |

### 1B. Reference numerics (reuse whole script as-is)

| Asset | Current location | Why |
|---|---|---|
| `scripts/torch_kitten.py` | as-is | **Single source of truth for correct numerical output.** Every ggml layer we write will be checked op-by-op against this. Without it the new port has no ground truth. |
| `scripts/real_calibration.py` | as-is | espeak-ng phonemizer + id-mapping + reference corpus. The IPA-to-id dict in this script already mirrors Swift Phonemizer.symbols. |
| `scripts/compare_wavs.py` | as-is | Spectral A/B harness. |
| `scripts/probe_istft_head.py` / `probe_upsamplers.py` / `probe_noise_path.py` | as-is | Op-level parity tests; we'll add a ggml counterpart to each. |
| `scripts/isolate_generator_bug.py` | as-is | End-to-end generator-only verification. Same method will localize any ggml-vs-torch drift. |

### 1C. Algorithmic knowledge (ports as logic, rewritten in C++)

Everything the Swift app does at the pipeline level has an exact line-by-
line analogue to port into C. The hard numerical bugs are all already
solved and documented in the code, so the new port starts from a known-
good reference rather than from upstream's original formulations:

| Lesson / fix | Where it lives | Why it matters |
|---|---|---|
| Zero `phase_jitter` + `uv_noise` in `compute_noise_contribs` | `scripts/torch_kitten.py` L736–747, `Sources/KittenApp/TTS.swift` L1263–1267 | Makes output bit-deterministic across backends. ggml port **must** do the same or it will diverge from torch reference and sound different per run. |
| Per-frame phase accumulation (short cumsum) | `scripts/torch_kitten.py` L715–731, `Sources/KittenApp/TTS.swift` L1253–1268 | fp32 long cumsum over 140k samples drifts audibly. Same risk in ggml: use per-frame formulation. |
| Tail-drop 3 frames before fade-out | `Sources/KittenApp/TTS.CoreML.swift` L255–268 | Click-at-end artifact kill. Decoder non-causal convs perturb last 3 real frames via zero padding; dropping them fixes it. Same fix applies verbatim. |
| 3 ms fade-in / 40 ms fade-out | `Sources/KittenApp/TTS.CoreML.swift` L260–261 | Smooths chunk boundaries. |
| 120 ms inter-sentence silence | `Sources/KittenApp/TTS.CoreML.swift` L118, `TTS.swift` L128 | Natural pause between streamed chunks. |
| Chunker maxLen=200 chars | `Sources/KittenApp/TTS.swift` L378 (and the comment block) | Keeps chunks under L=400 phoneme and N=1024 frame ceilings. |
| Text chunker: split on `.!?;:`, preserve punctuation, split oversize sentences on `, ` | `TextChunker.chunk` in TTS.swift | Good UX heuristic to port. |
| Overflow-trim durations tail-ward when `totalFrames > nBucket` | `TTS.CoreML.swift` L202–212 | Only used when a single chunk somehow exceeds N_max. Port the pattern. |
| Bucketing (fixed-shape L and N) vs dynamic shapes | `TTS.CoreML.swift` static lets | ggml supports dynamic shapes natively, so we can skip bucketing! See §3 design choice. |
| Speak/Stop generation counter | `KittenApp.swift` `AudioPlayer.beginGeneration` / `playChunk(generation:)` | UI pattern for reliable stop. |
| Warmup gating modelReady | `KittenApp.swift` `loadBackend` | UX pattern; applies to any backend with compile-time-per-shape cost. Less relevant for ggml (no ANE compile) but useful for metal shader warm-up. |

### 1D. Does NOT transfer

- `Sources/KittenApp/Resources/coreml/*.mlpackage` — CoreML-specific.
- MLX-swift integration, `@preconcurrency import CoreML` — Apple-only.
- SwiftUI KittenApp UI — Android would get a Jetpack Compose UI, CLI
  would get a different front-end.
- `scripts/convert_to_coreml.py`, `calibrate_and_quantize.py`,
  `build_multifunction.py`, `quantize_coreml.py` — CoreML-specific.

## 2. New-repo layout proposal

```
kittens-ggml/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── external/
│   └── llama.cpp/                 # git submodule, --depth=1, master
├── src/
│   ├── kittens_arch.cpp           # ggml graph builder (new llm_arch)
│   ├── kittens_arch.h
│   ├── phonemizer/                # vendored from current CEPhonemizer/
│   │   ├── phonemizer.cpp
│   │   ├── phonemizer.h
│   │   ├── ipa_table.h
│   │   └── rule_parser.h
│   ├── text_chunker.cpp           # port of TextChunker / TextPreprocessor
│   ├── text_chunker.h
│   ├── pipeline.cpp               # length-reg, tail-drop, fades, gap
│   ├── pipeline.h
│   ├── kittens.h                  # public C API (see §3)
│   └── kittens.cpp
├── cli/
│   └── kittens_tts.cpp            # libsndfile / raw PCM CLI
├── android/                       # JNI + Jetpack Compose shell
├── python/
│   ├── convert_kittentts_to_gguf.py
│   └── validate_gguf.py           # round-trip vs torch_kitten.py
├── tests/
│   ├── parity/
│   │   ├── test_bert.cpp
│   │   ├── test_lstm.cpp
│   │   ├── test_adain.cpp
│   │   ├── test_snake.cpp
│   │   ├── test_istft.cpp
│   │   └── test_end_to_end.cpp
│   └── corpus/                    # copied from Resources/prompts/
├── assets/
│   ├── en_rules
│   ├── en_list
│   ├── voices.bin                 # re-encoded from voices.safetensors
│   └── kitten_tts_nano_v0_8.gguf
└── scripts/
    ├── torch_kitten.py            # copied, frozen reference
    ├── compare_wavs.py
    └── probe_*.py
```

## 3. Architectural design choices

### 3A. One GGUF file, not two

llama.cpp's `tools/tts` is two-model (LLM + vocoder) because OuteTTS is
autoregressive. KittenTTS is non-AR, so we ship **one** `.gguf` holding
the whole thing (TextStage + GeneratorStage tensors share the file,
just like our multifunction `.mlpackage`). Similar weight dedup story.

### 3B. Dynamic shapes, no buckets

ggml supports runtime-resizable graph dimensions. The bucketing we had
to do for CoreML (L ∈ {16, 32, 64, 128, 400}, N ∈ {128, 256, 512, 1024})
was a workaround for CoreML's flex-shape runtime bug — it's not needed
in ggml. We build the graph per call with the actual L and
actual nFrames. Simpler, fewer files, zero truncation risk.

### 3C. No activation quantization in scope for v1

Weight quantization via llama.cpp's Q4/Q5/Q6/Q8/IQ is trivial (GGUF
metadata flag). Activation quantization was the hard pain point in the
CoreML work. Skip it for v1; ship fp32 and int8-weight variants only.

### 3D. Public C API (stable ABI)

```c
// kittens.h — what external Swift / Kotlin / Python binds to
typedef struct kittens_context kittens_context;

kittens_context * kittens_init(const char * gguf_path,
                               const char * en_rules_path,
                               const char * en_list_path,
                               const char * voices_bin_path);
void              kittens_free(kittens_context * ctx);

// Returns PCM float samples at 24 kHz mono. Caller owns the returned
// buffer; free via kittens_audio_free. Zero-noise, deterministic.
typedef struct {
    float *   samples;
    uint64_t  num_samples;
} kittens_audio;

kittens_audio kittens_speak(kittens_context * ctx,
                            const char * text,
                            const char * voice_id,    // e.g. "Kiki"
                            float        speed);
void          kittens_audio_free(kittens_audio * a);

// Streaming variant: callback invoked once per sentence-chunk.
typedef void (*kittens_chunk_cb)(const float * samples, uint64_t n,
                                 void * user);
int kittens_speak_streaming(kittens_context * ctx,
                            const char * text, const char * voice_id,
                            float speed,
                            kittens_chunk_cb cb, void * user);
```

Swift binds through the bridging header we already use for CEPhonemizer.
Android binds via JNI. Python binds via `ctypes` / `cffi`.

## 4. Phased execution plan

### Phase 1 — Converter + numeric ground truth (1 week)

- [ ] Bootstrap new repo, add `llama.cpp` as `--depth=1` submodule.
- [ ] Copy `scripts/torch_kitten.py` verbatim (freeze as v1 reference).
- [ ] Write `python/convert_kittentts_to_gguf.py`:
  - Load safetensors with our existing `WeightBag.load`.
  - Map every tensor to a GGUF name matching our forthcoming C++
    graph-builder's expectations.
  - Encode architecture metadata (voice-count, style-dim, bucket sizes
    if any, STFT kernel size, etc.) as GGUF key-value pairs.
  - Emit `kitten_tts_nano_v0_8.gguf` (fp16 target, ~40 MB).
- [ ] Write `python/validate_gguf.py`: reads the .gguf back in Python
  using ggml bindings / manual parser, feeds to a reference-only
  pipeline (still torch), asserts numerical identity with
  `torch_kitten.py` on a fixed input.
- [ ] Publish the .gguf to HuggingFace as `kitten-tts-nano-v0.8-gguf`
  so other projects can consume.

**Gate**: byte-identical weights round-trip via GGUF.

### Phase 2 — C++ graph builder, op-by-op (2 weeks)

For each component, write the ggml graph builder, then write a parity
test against the corresponding Python component. All tests compare to
`torch_kitten.py` outputs.

1. **BertStack** (Albert shared 12-layer) — structurally similar to
   llama.cpp's existing LLaMA graph. Easy. Borrow the attention +
   LayerNorm helpers from llama.cpp's source. Test against
   `torch_kitten.TextStage.bert`.

2. **ONNX bidirectional LSTM (unrolled)** — write a reusable
   `kittens_bidir_lstm(ctx, x, mask, W, R, B, H)` helper. Unrolled per
   timestep with ggml_sigmoid / ggml_tanh / ggml_mul_mat / ggml_add.
   Two directions in sequence, concat. Test with same inputs as our
   existing `scripts/verify_coreml_vs_torch.py` uses.

3. **AdaIN 1D** — ggml_group_norm(G=C) + `h = style @ fcW + fcB`,
   split into γ/β, `normed * (1 + γ) + β`. Existing ggml primitives.

4. **Snake activation** — build as `x + (1/α) · sin(α·x)²`. Needs
   trainable α tensors per block. Already in GGUF from converter.

5. **AdaINResBlock1D + AdaINResBlockHiFiGAN** — compose from the above.

6. **Predictor text encoder, acoustic text encoder** — LSTMs + AdaIN +
   conv1d. All pieces already done.

7. **Duration predictor** — linear head, sigmoid, sum-of-50 logits
   trick, round to int, max(1, ...). Identical to Swift logic.

8. **Length regulation** — not a ggml op; build alignment matrix in C,
   then ggml_mul_mat. Or even simpler: cpu-side expansion into a
   pre-sized tensor, feed into the generator graph.

9. **F0 / N paths** — AdaINResBlock1D chains with upsample=True variants
   via `ggml_conv_transpose_1d`. Test against torch probes.

10. **Noise path (compute_noise_contribs with cumsum fix + zero noise)**
    — per-frame phase formulation from day one. No dither.

11. **Decoder + generator pipeline** — conv_transpose upsamples +
    reflection_pad_left + resblocks. `pad_reflect_1d` exists, but it's
    symmetric — we need left-only. Small custom ggml helper (slice +
    flip + cat).

12. **iSTFT head** — conv_transpose_1d(stride=5) for real/imag + trim.
    Straight port.

13. **Pipeline**: length-regulation between text and generator, tail-
    drop 3 frames, 3 ms / 40 ms fades, 120 ms inter-sentence gap.

**Gate**: end-to-end cosine similarity ≥ 0.999 against
`torch_generate.py` on a 16-sentence corpus, per bucket shape class.

### Phase 3 — CLI + packaging (1 week)

- [ ] `cli/kittens_tts.cpp` reading stdin / `-t`, writing WAV via
  libsndfile or manual RIFF.
- [ ] CMake packaging — one `kittens_tts` static lib + one
  `kittens_tts_cli` binary. Cross-compile targets: linux-x86_64,
  linux-arm64, android-arm64 (via NDK), macOS-arm64, iOS-arm64.
- [ ] GitHub Actions: build + run parity tests + produce release
  artifacts per platform.
- [ ] README with usage, numbers, and a matrix of op status.

### Phase 4 — bindings (0.5 week each, as needed)

- [ ] Swift bridge: mirror `TTS.CoreML.swift` structure, link static lib.
- [ ] Android AAR with JNI wrapper + Kotlin API.
- [ ] Python wheel for non-KittenApp consumers.

## 5. Testing & numerical parity

Non-negotiable. Every op test starts from `torch_kitten.py` output as
ground truth. Pass criteria:

- **Per-op**: `cos >= 0.9999` AND `max_abs_err <= 1e-4` on the test
  inputs that exercise realistic ranges (style samples, real phoneme
  sequences, actual f0 ranges).
- **End-to-end**: on a 16-sentence corpus, 24 kHz audio output must
  satisfy `cos >= 0.999` against torch reference.
- **Regression**: CI runs the corpus on every PR and diffs the
  resulting WAVs against a checked-in reference at 8-bit quantization.
- **Fuzz**: random text inputs, assert no crashes, no NaNs.

The op-probe scripts we have (`probe_istft_head.py`,
`probe_upsamplers.py`, `probe_noise_path.py`) become the Python half of
each parity test; a new `test_*.cpp` is the ggml half that reads the
same input tensors from .npy and produces a matching output.

## 6. Effort breakdown (revised)

| Phase | Work | Estimated effort |
|---|---|---:|
| 1 | Converter + GGUF validation | 5–7 days |
| 2 | Graph builder, per-op parity tests | 10–14 days |
| 3 | CLI, packaging, CI | 5–7 days |
| 4 | Swift / Android / Python bindings | 2–4 days each |
| — | Numerical-parity debug buffer | +5–10 days |

**Realistic**: ~5–8 weeks elapsed for one engineer working steadily,
with the first usable end-to-end demo around week 3.

## 7. What would make this faster

- If upstream KittenTTS (HuggingFace `KittenML/kitten-tts-nano-0.8`)
  lands an official GGUF, we skip Phase 1.
- If anyone publishes a ggml StyleTTS2 port first, we can fork it and
  reuse the AdaIN/snake/iSTFT graph builder.
- If we decide to drop the BERT text encoder and use espeak-ng phoneme
  IDs directly (no text-encoder LSTM stack), we save ~30% of Phase 2.
  But that changes the model, so out of scope here.

## 8. Risks & open questions

1. **LSTM unrolling perf**: 12 layers × 2 directions × ~130 timesteps
   will hit graph-build overhead on ggml. May want a packed LSTM helper
   or ask upstream to add `GGML_OP_LSTM`. Mitigation: benchmark early
   (end of Phase 2.2).
2. **Metal-embed build size**: `GGML_METAL_EMBED_LIBRARY=ON` bakes the
   default.metallib into the static lib. Size addition ~6 MB. Acceptable.
3. **Android NDK + Metal**: there's no Metal on Android. We rely on
   ggml's Vulkan backend there. Vulkan backend in ggml is maturing but
   less battle-tested than Metal/CUDA/CPU. Early benchmark on a
   representative device recommended.
4. **Phonemizer regression risk**: CEPhonemizer is our rule-based
   G2P. Tested on English. Other languages not in scope. Keep that
   clear in README.
5. **ABI stability**: if `kittens.h` evolves, bindings break. Version
   the C API from day one with `KITTENS_ABI_VERSION` macro.

## 9. Recommendation

Yes, this is doable as a separate repo. The current Kittens work
**saves ~40% of the effort** — mainly in the Python reference, the
phonemizer, the sample prompts, and most importantly the hard-won
numerical-correctness lessons (zero noise, per-frame cumsum, tail-drop,
chunking, fades). Without those we'd be rediscovering three audible
bugs from scratch in C++. With them, Phase 2 is a mechanical
port-and-verify against a ground truth we already trust.

Open the new repo, add llama.cpp as a `--depth=1` submodule, copy
Phase 1 inputs, start on the converter. Pause before committing to
Phase 2 until GGUF round-trip is bit-identical.
