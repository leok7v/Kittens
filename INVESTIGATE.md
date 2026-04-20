# KittenTTS CoreML — Investigation and Plan

Status file meant to survive context compaction. If a fresh AI session picks
up work here, read this document end-to-end before touching code.

---

## 0. Current state snapshot (2026-04-19)

Both MLX and CoreML backends are shipping and sound clean on short and long
text. The original 720 Hz CoreML tonal artifact is gone. Key changes since
the historical sections below were written — do NOT relitigate these:

- **Zero dither**: `phase_jitter` / `uv_noise` in the noise path are
  forced to zero in BOTH `scripts/torch_kitten.py`
  (`compute_noise_contribs`) AND `Sources/KittenApp/TTS.swift`
  (`computeNoiseContribs`). Reason: torch and CoreML (and MLX) draw from
  independent RNGs; keeping those terms makes output diverge per run
  and per backend. noise_std was 0.003 and phase_jitter was per-harmonic
  — both imperceptible in speech. Zeroing them makes output bit-reproducible.
- **Per-frame phase cumsum**: the long cumsum over ~144k samples accumulates
  fp32 drift audibly; replaced by a short (≈480-element) per-frame cumsum
  plus within-frame linear extrapolation. Applied to both
  `compute_noise_contribs` (torch) and `computeNoiseContribs` (MLX).
  Isolated probe: without fix, torch-vs-CoreML cos = 0.44; with fix,
  cos = 0.999985. Pipeline-level cos went 0.47 → 0.98.
- **End-click fix**: `tailDropFrames = 3` in `TTS.CoreML.swift
  speakOneChunk`. Decoder non-causal convolutions bleed zero-padding
  artifacts into the last ~3 real frames; dropping those frames before
  fade-out kills the click. Fades are 3 ms in / 40 ms out.
- **Chunker maxLen = 200 chars** in `TextChunker.chunk` in TTS.swift.
  Single sentences longer than 200 chars are split on `", "`. Keeps
  chunks under L=400 phoneme truncation and N=1024 audio-frame
  overflow-trim (which compressed trailing-phoneme durations and made
  long sentences "speed up at the end" — that bug is now unreachable).
- **Inter-sentence silence**: 120 ms of zeros between streamed chunks
  in both `TTS.swift` and `TTS.CoreML.swift`. Natural comma-like pause.
- **CoreML export**: after re-quantization with real calibration and
  the numerical fixes above, we ship `fp32`, `fp16`, `int8w`. `int8wa`
  is parked — see §12 of this document. Winner config is
  `coreml/int8w/all` (ANE): warm RTF ~25×, ~50 MB RAM.
- **UI**: Speak/Stop single toggle button (no layout shift), Stop
  cancels inference AND drops any late chunks via an AudioPlayer
  generation counter, Prompt picker with 16 presets (4 built-in + 13
  stories from `misc/`), persistent @AppStorage for voice/backend/
  variant/compute/prompt, default prompt on fresh install is the
  "Long paragraph" stress test. Voice list ordered Hugo, Luna, Kiki,
  Leo, Bella, Jasper, Bruno, Rosie (subjective cleanness).
- **iOS fixes**: AVAudioSession set to `.playback` + `.spokenAudio` +
  `.duckOthers` so audio survives the silent-switch; `minWidth 520`
  floor is `#if os(macOS)` only; stale `mlx.metallib` codesign issue
  cured by gating `installMetalLib()` to non-.app bundles.
- **ANE warmup**: `coreMLTTS.warmUpAll(variant:compute:)` walks every
  L and N bucket; loaded during `loadBackend` so `modelReady = true`
  only flips after compile is done. Variant/compute/backend switches
  re-gate with a "Warming up…" status.

### Next direction (pending decision)

`LLAMA.BACKEND.md` at repo root is the plan for a **separate** new repo
that ports KittenTTS to llama.cpp/ggml for portable Linux/Android
targets. Not started. Current Kittens repo saves ~40% of that work
(reference numerics in `scripts/torch_kitten.py`, CEPhonemizer C++,
test corpus, all the hard-won numerical fixes listed above). Read
`LLAMA.BACKEND.md` before doing anything on the ggml side.

---

## 1. Goal

Provide an on-device TTS app for Apple platforms (macOS, iOS, eventually
watchOS) that uses the same KittenTTS model as `KittenML/KittenTTS-swift`
(checked into this repo for reference), but runs on Apple-native inference
paths (CoreML and/or MLX) instead of ONNX Runtime, while **matching
ONNX Runtime's performance and RAM footprint** on a per-backend basis.

Nonnegotiable constraints:

- **No `onnxruntime` dependency.** The whole point is to run without ORT.
- **Keep MLX backend for A/B reference.** MLX is a hand-port of the same
  model (see `Sources/KittenApp/TTS.swift`, ~1300 lines, verified against
  ONNX Runtime output — see `scripts/verify_torch_vs_onnx.py`).
- **Quality must match ORT's output**, within documented quantization drift.
- Model weights (`kitten_tts_nano_v0_8` INT8, ~25 MB on disk) stay the
  canonical "model" — we ship or rebuild from that.

---

## 2. Baseline — what ORT achieves

Reference impl lives at `KittenML/KittenTTS-swift/`. Key facts:

- Dependency: `onnxruntime-swift-package-manager` ≥ 1.20.0.
- Loads the 25 MB INT8 ONNX file once (`kitten_tts_nano_v0_8.onnx`).
- Runs on CPU only (`session.run(...)`, no CoreML EP — code comment says
  "CoreML EP is intentionally excluded to avoid shape-inference errors").
- Dynamic shapes — one graph, any input length. No bucketing.
- `DynamicQuantizeLSTM` (a Microsoft custom op in the ONNX) runs **INT8
  at inference**; activations are quantized on the fly via
  `DynamicQuantizeLinear` inserted around every `MatMulInteger` and
  `ConvInteger`.
- `TextPreprocessor` splits long text into chunks of ≤ 400 tokens
  (`KittenTTSConfig.maxTokensPerChunk`, floor 50).
- Voice selection via `VoiceEmbedding.slice(forTextLength:)` — picks row
  `min(len, 399)` from the 400×256 embedding table.
- Measured on M-series macOS: **~265 MB RAM, ~1 s preload, RTF
  competitive with our impl**.

Model variants KittenML ships (from `KittenModel.swift`):

| variant       | weights | size     |
|---------------|---------|----------|
| `nano`        | fp32    | ~56 MB   |
| `nanoInt8`    | int8    | ~25 MB   |
| `micro`       | fp32    | ~41 MB   |
| `mini`        | fp32    | ~80 MB   |

We only use nano (fp32) and nanoInt8 in this repo.

---

## 3. Current state of our implementation (before this investigation)

Committed on branch `main`. Key pieces:

### Project layout

- `Kittens.xcodeproj` — Swift 6 app target `KittenApp`, macOS 15 + iOS 18.
  Uses `PBXFileSystemSynchronizedRootGroup` so everything under
  `Sources/KittenApp/` is auto-picked-up.
- `Sources/KittenApp/TTS.swift` — **MLX backend** (≈1300 lines). Hand port
  of `kittenForward` with custom conv1d/conv_transpose1d/LSTM in MLX.
  Verified against ORT output.
- `Sources/KittenApp/TTS.CoreML.swift` — **CoreML backend**, consumes
  traced `.mlpackage` files produced by our Python scripts.
- `Sources/KittenApp/CEPhonemizer/` — C++ IPA phonemizer (same as in
  `KittenML/KittenTTS-swift/Sources/CEPhonemizer/`). Exposed to Swift
  through `KittenApp-Bridging-Header.h`.
- `Sources/KittenApp/Resources/` — flattened resources:
  - `nano/` — safetensors + voices + phonemizer data.
  - `coreml/` — 7 `.mlpackage` files (4 text buckets + 3 generator
    buckets), all INT8 weights / fp16 activations. Xcode compiles them
    to `.mlmodelc` at build time.
  - `mlx.metallib` — for MLX.
- `Vendor/mlx-swift/` — local SwiftPM dep for MLX/MLXNN.
- `scripts/.venv/` — Python env (gitignored) with `torch 2.11`, `coremltools 9.0`,
  `onnxruntime`, `onnx2torch`, `numpy`, `safetensors`.

### The CoreML pipeline we built

See `scripts/torch_kitten.py` + `scripts/convert_to_coreml.py` +
`scripts/quantize_coreml.py`. High level:

1. **PyTorch port** of `kittenForward` in `scripts/torch_kitten.py`
   (~1000 lines), split into **two stages** to handle the dynamic
   length-regulation step in Swift:

   - `TextStage(input_ids, style256, attention_mask)` → `(prosody_ncl,
     text_features_ncl, dur_sig)`. Fixed on axis `L` (phoneme count).
   - `GeneratorStage(prosody_lr, text_lr, style256)` → `waveform`.
     Fixed on axis `nFrames` (after length regulation).
2. Swift side: compute durations from `dur_sig`, build alignment matrix,
   expand `prosody_ncl` / `text_features_ncl` to `nFrames`, run
   `GeneratorStage`.
3. coremltools 9 dropped the ONNX→CoreML frontend entirely; our path is
   PyTorch → `torch.jit.trace` → coremltools MIL → `.mlpackage`.
4. Ported LSTM: started as Python-unrolled masked LSTM (to handle pad
   tokens correctly), then reverted to `torch.nn.LSTM` wrapper with
   ONNX→PyTorch gate reordering (iofc→ifgo) for 10× speed.
5. Quantization done post-hoc with
   `ct.optimize.coreml.linear_quantize_weights` (weights → INT8 on disk).

### Known problems (why we're investigating)

| dimension   | ORT    | Ours (as of commit) | Gap     |
|-------------|--------|---------------------|---------|
| RAM         | 265 MB | **900 MB**          | 3.4×    |
| First-speak | instant | ~1 s (warmup we added) | slower |
| Disk        | 25 MB  | 60 MB (7 mlpackages INT8) | 2.4× |
| On ANE      | N/A    | nominally, but ANE does fp16, not INT8 | — |

**Root-cause breakdown of the 635 MB extra RAM:**

| cause | RAM cost |
|---|---|
| Our 7 CoreML buckets eagerly loaded + ANE-warmed (I added this to hide JIT cost; trades memory for first-speak latency) | ~300–400 MB |
| MLX runtime still resident (safetensors cache, Metal kernels) even after `unload()` is called on backend switch | ~100–200 MB |
| Our "INT8" CoreML models **store weights INT8 on disk but expand to fp16 in RAM at load time** — this is how `linear_quantize_weights` works. ANE multiplies fp16, not INT8. Per-bucket weights ≈ 14 MB fp16, not 3.5 MB INT8. | ~20–40 MB |
| CoreML runtime overhead per loaded bucket (ANE compute descriptors, activation buffers, JIT state) | ~50–100 MB |
| coremltools MIL graph materialisation adds shape metadata and constants that ORT's proto-based graph skips | ~30–50 MB |

The ORT impl keeps its weights INT8 in RAM because `DynamicQuantizeLSTM`
and `MatMulInteger` kernels multiply INT8 × INT8 → INT32 directly. ANE
cannot do that, so CoreML's `linear_quantize_weights` dequantizes at
load time. **That is the single biggest architectural difference.**

---

## 4. Decisions agreed so far

- **Keep MLX backend** in-app for A/B reference. No removal.
- **Do not add ORT.** No `onnxruntime` dep, period.
- **Ship multiple CoreML quantization variants for A/B**:
  - `fp32`   — weights fp32, activations fp16.
  - `fp16`   — weights fp16, activations fp16. ANE-optimal.
  - `int8w`  — weights INT8 (expanded to fp16 at load time), activations fp16.
  - `int8wa` — weights INT8 + activations INT8 via
    `linear_quantize_activations`, CPU-only at runtime. Matches ORT's
    shape+precision 1:1.
- **UI pickers** (see §6 below): Backend, Variant, Compute.
- **Do not prune variants yet.** Ship all, pick winner later.
- **Disk usage isn't yet optimised.** We'll have ~290 MB of CoreML
  artefacts in dev builds. Cleanup is a later pass.

---

## 5. Plan — A → B → C with defined fallbacks

Goal: single dynamic-shape CoreML model per variant per stage, matching
ORT's "one graph, any length" architecture.

### Plan A (attempt first) — `torch.export` + truly dynamic shapes

`torch.jit.trace` bakes specific shapes into MIL at trace time (`reshape(1, L, 128)`
becomes `reshape(1, 32, 128)` constant). That's why our earlier `RangeDim`
flex attempt converted but failed at runtime for any L ≠ trace-time L.

`torch.export` (PyTorch 2.2+) tracks symbolic dimensions through the entire
graph via `torch.fx` + shape-symbol propagation. Coremltools 7+ accepts
`ExportedProgram` objects directly.

Required changes for Plan A:
1. **Audit `scripts/torch_kitten.py`** for every operation that reads a
   concrete dim or reshapes to a Python int:
   - `input_ids.size(1)` and similar shape reads → use `input_ids.shape[1]`
     (SymInt) and propagate through arithmetic.
   - Every `reshape(1, L, 128)` → `reshape(1, -1, 128)`.
   - Every `torch.arange(L, ...)` → use tensor-level arange from shape.
   - `range(L)` in Python loops → OK only if we can unroll; otherwise
     flatten to tensor ops.
2. Write `scripts/convert_dynamic.py` that:
   - Builds `TextStage` and `GeneratorStage`.
   - Uses `torch.export.export(stage, args, dynamic_shapes={...})` with
     `Dim("L", min=16, max=400)` (text) and `Dim("N", min=32, max=1024)`
     (generator).
   - Feeds the ExportedProgram to `ct.convert(...)`.
   - Saves one `.mlpackage` per variant per stage.
3. Verify with a **runtime shape sweep**: invoke the same `.mlpackage`
   at L ∈ {16, 32, 64, 128, 256, 400} and check outputs match torch at
   the same shape.

**Success criterion**: one `.mlpackage` per (stage × variant) that
runs at any L in [16, 400] without error. If that's achieved, plan A
is done and B/C are unused.

**Budget**: up to ~40 k tokens. If we hit an impassable issue (e.g.,
coremltools doesn't support some op in dynamic-shape mode, or certain
attention patterns don't generalise), document the exact failure at
the end of this file under §9 "Plan A failure notes" and move to B.

### Plan B (current path) — Multifunction `.mlpackage`

Supported from coremltools 7+ / iOS 17+ / macOS 14+. One `.mlpackage`
can contain multiple functions, each compiled for a specific shape,
**sharing weights at the file level**. Swift selects a function at
load time via `MLModelConfiguration.functionName`.

If we end up here:

- **Shapes to ship**: `L ∈ {32, 64, 128, 256, 400}`, `N ∈ {64, 128, 256, 512, 1024}`.
- One `.mlpackage` per stage per variant. Weights stored once; each
  shape contributes a small function definition block on top.
- Swift's existing `textBucket(_:)` / `generatorBucket(_:)` logic stays,
  but instead of loading a different `.mlpackage` per bucket, it
  instantiates the right function from one file.
- Each function still pays its own ANE JIT cost on first call, but the
  weight-loading cost happens once per file (across all functions).

**Success criterion**: ≤ 10 `.mlpackage` files total for (2 stages × 4
variants = 8, with some variants possibly collapsed); per-variant RAM
in the 150–250 MB range.

### Plan C (last resort) — trimmed per-bucket `.mlpackage` set, lazy-loaded

Our current architecture but with:
- Only 2 text buckets: L ∈ {128, 400}.
- Only 2 generator buckets: N ∈ {256, 1024}.
- 4 variants.
- → 16 `.mlpackage` files.
- No eager preload; lazy on first use per bucket.

This is guaranteed to work and gets us to ~300–400 MB RAM.

---

## 6. Variants matrix & UI

| variant | weights | activations | compute typically on | matches ORT |
|---|---|---|---|---|
| `fp32`   | fp32 | fp16 | ANE/GPU | — |
| `fp16`   | fp16 | fp16 | ANE/GPU | — |
| `int8w`  | int8 (→ fp16 at load) | fp16 | ANE/GPU | — |
| `int8wa` | int8 | int8 | CPU (ANE can't int8 matmul) | **yes** |

Swift runtime picks compute unit independently via `MLComputeUnits` setting:

- `.all`      — CoreML chooses (ANE/GPU/CPU by op).
- `.cpuOnly`  — forces CPU. With `int8wa` this is effectively the only
  option that makes sense.

### UI additions in `Sources/KittenApp/KittenApp.swift`

Three pickers below the text editor, stacked vertically:

```
Voice    [Leo ▾]                              (existing)
Backend  [ MLX | CoreML ]                     (existing, segmented)
Variant  [ fp32 | fp16 | int8w | int8wa ]    (new, segmented, disabled when Backend=MLX)
Compute  [ All  | CPU ]                      (new, segmented, disabled when Backend=MLX)
Speed    [----●-------] 1.00×                 (existing)
```

All four should persist via `@AppStorage` (`voice`, `backend`, `variant`,
`compute`) — no separate sessions.

Log line prefix changes to include variant + compute so A/B rows are
unambiguous in the metrics log:

```
CoreML/int8w/ANE   chunk  phonemes=78 L=128 N=256  text 61ms gen 153ms  audio 4.78s  RTF 22.3x
```

---

## 7. Calibration for `int8wa`

`ct.optimize.coreml.linear_quantize_activations(model, config, sample_data=...)`
needs a batch of representative inputs so coremltools can observe
activation ranges at every quantizable op.

Plan:
1. Source: bundled Harvard-sentence corpus (public domain, phonetically
   balanced, ~720 short English sentences). Download/cache in
   `scripts/calibration/`.
2. Script: `scripts/calibrate_and_quantize.py`:
   - Phonemize 64 random sentences via Python-side wrapper around the
     CEPhonemizer C library (or duplicate-port) — or fall back to
     `torch_kitten.py` + random valid token sequences if linking C is
     annoying.
   - For `TextStage`: produce `(input_ids, style, attention_mask)`
     tuples where `input_ids` = phonemized sentence (padded to bucket
     L if Plan C), `style` = random row of `voices.npz`,
     `attention_mask` = 1s on real, 0 on pad.
   - For `GeneratorStage`: run each input through the fp32 `TextStage`,
     compute durations and length-regulated tensors (same logic as
     Swift side), produce `(prosody_lr, text_lr, style)` calibration
     tuples.
   - Save calibration tuples to disk so quantization is reproducible.
3. Feed to `linear_quantize_activations`. Save quantized `.mlpackage`
   variants.

64 samples is coremltools' documented sweet spot for this model size.

---

## 8. Execution roadmap

Ordered task list, each a checkpoint. Mark done as we progress.

- [ ] Write `INVESTIGATE.md` (this file) + reference from `README.md`.
- [ ] Audit `scripts/torch_kitten.py` for shape-constant bakes; refactor
      to use symbolic shape reads everywhere.
- [ ] Write `scripts/convert_dynamic.py` — `torch.export` + coremltools
      dynamic-shape conversion for TextStage and GeneratorStage at fp32.
- [ ] Sweep-test: load the converted model, run at L ∈ {16, 32, 64, 128,
      256, 400}, verify outputs numerically match per-shape torch runs.
- [ ] If sweep fails → append root cause + failed shape(s) to §9, move
      to Plan B.
- [ ] fp16 variant: rerun conversion with `compute_precision=FLOAT16`.
- [ ] int8w variant: apply `linear_quantize_weights` to the fp32/fp16
      converted model.
- [ ] Calibration corpus: download Harvard sentences; build calibration
      tuples via the calibrate script.
- [ ] int8wa variant: `linear_quantize_activations` with calibration.
- [ ] Swift loader: replace bucket-array lookup with "load one model per
      variant; feed any length input". Remove bucket dict.
- [ ] UI: add Variant + Compute pickers, persist via @AppStorage,
      thread through to `speak()`. Update log prefix.
- [ ] Rebuild app; test on macOS at every (variant × compute) combo.
      Record RAM / RTF / audio-duration / audio-quality-listen for each.
- [ ] Test on iPhone — same matrix, compare to macOS numbers.
- [ ] Update metrics comparison table at the end of this file.
- [ ] Decide winner, plan cleanup pass (separate PR).

---

## 9. Plan A failure notes

**Outcome:** Plan A failed at `torch.export.export()` before even reaching
coremltools. Root cause is **`torch.nn.LSTM`'s backend decomposition
specializes the sequence-length dim to its concrete trace-time value**.

**Exact error:**

```
torch._dynamo.exc.UserError: Constraints violated (L)!
  - You marked L as dynamic but your code specialized it to be a constant (64).
```

**Debug output showed:**

```
create_symbol s53 = 64 for L['attention_mask'].size()[1] [16, 400]
eval Eq(s70, s53) [guard added] (_subclasses/fake_impls.py:1503 in infer_size)
eval Eq(s53, 64) [guard added] (_ops.py:910 in decompose)      ← the bad one
set_replacement s53 = 64 (range_refined_to_singleton) VR[64, 64]
```

The `(_ops.py:910 in decompose)` guard is added during ATen op
decomposition — stack trace pointed to `scripts/torch_kitten.py:310`
which is `out, _ = self.lstm(x)` inside `ONNXBidirLSTM.forward`. So
`nn.LSTM` is the culprit; its ATen decomposition produces a
specialization constraint that pins `L = trace-time L`.

Workarounds we considered, none clean:

- **Use `torch._VF.lstm` directly** — same decomposition path, same
  specialization.
- **Manual LSTM with `F.lstm_cell` + Python loop over seq** — loops
  over dynamic `SymInt` ranges aren't supported by `torch.export` in
  stable PyTorch 2.11. Would require `torch.scan` (experimental).
- **TorchScript (`torch.jit.script`)** instead — same fundamental
  issue with LSTM decomposition, plus scripting would require
  rewriting a lot of the port.
- **`torch.export` with `strict=False`** — already the default path
  (`_non_strict_export`). No change.
- **Upgrade PyTorch** — no published release fixes this for nn.LSTM
  with dynamic seq length.

**Conclusion:** a single truly-dynamic CoreML model is **not achievable
today** via `torch.export + coremltools 9` without rewriting the LSTMs
to a form PyTorch's export machinery can symbolically trace. That's a
multi-day effort of uncertain outcome.

**→ moved on to Plan B.**

Artefacts left on disk:

- `scripts/convert_dynamic.py` — the working Plan A script. Left in
  place in case a future PyTorch release fixes the nn.LSTM decomp.
- `scripts/models/dynamic/` — not populated (conversion never reached
  the .mlpackage save step).

---

## Plan B execution progress (live)

**Multifunction API confirmed working.** `scripts/build_multifunction.py`
uses `coremltools.models.utils.MultiFunctionDescriptor` +
`save_multifunction` to merge per-bucket `.mlpackage` files into one
multi-function file. At runtime we load with `function_name="L_128"` (etc.)
and the right shape specialization is used.

**Weight dedup is effective:**

| input | size each | count | total input | multifunction output |
|---|---|---|---|---|
| `kitten_text_L{16,32,64,128}.mlpackage` int8w | ~6.5 MB | 4 | 26 MB | **7.1 MB** |
| `kitten_generator_N{128,256,512}.mlpackage` int8w | ~7.7 MB | 3 | 23 MB | **8.5 MB** |

Cross-function weight overhead is <10% of a single bucket's size.
Dedup clearly happens at merge time.

Runtime sanity check (Python + coremltools):
```
L=16  shapes=['(1, 128, 16)',  '(1, 16, 50)',  '(1, 256, 16)']
L=32  shapes=['(1, 128, 32)',  '(1, 256, 32)', '(1, 32, 50)']
L=64  shapes=['(1, 128, 64)',  '(1, 256, 64)', '(1, 64, 50)']
L=128 shapes=['(1, 128, 128)', '(1, 128, 50)', '(1, 256, 128)']
```
All four function shapes produce correct outputs from the one file.

**Remaining Plan B tasks:**

- [ ] Convert + add L=400 text bucket and N=1024 generator bucket (if
      we want ORT's 400-token ceiling coverage).
- [ ] Convert fp32 per-bucket mlpackages (`--variant fp32` on
      `convert_to_coreml.py` — just `compute_precision=FLOAT32` flag).
- [ ] Convert fp16 per-bucket mlpackages
      (`compute_precision=FLOAT16`).
- [ ] Re-quantize `int8w` on top of fp16 conversion (not fp32) for
      ANE-native compression; may change size/quality slightly vs
      current.
- [x] `int8wa` calibration — **working after two attempts**.
      `scripts/calibrate_and_quantize.py` now produces all 9 int8wa
      mlpackages. Two changes vs. the first-attempt blockers:

      1. Source the calibration from the **fp16** converted model, not
         fp32. coremltools' `linear_quantize_activations` emits fp16
         `scale` values; an fp32 source tensor triggers a dtype
         validator (`scale has dtype fp16 whereas input has dtype fp32`).

      2. Use `op_type_configs` with `global_config=None` and an explicit
         whitelist (`linear`, `matmul`, `conv`) instead of trying to
         quantize everything. Filters out `gather` (embedding lookup
         with int32 indices — the first-attempt blocker) and leaves
         softmax / layernorm / lstm in fp16.

      Sizes: text 6.6 MB/bucket, generator 7.8 MB/bucket. Multifunction
      merged: `kitten_text_int8wa.mlpackage` = 7.9 MB,
      `kitten_generator_int8wa.mlpackage` = 9.7 MB.

      Calibration cost: N=32 samples per text bucket + N=16 samples per
      generator bucket (reduced from 32 for N≥512 to keep runtime
      manageable; N=1024 × 32 samples took ~30 min and was killed).
- [ ] Merge all variants into multifunction files. Final artifact set
      (to copy into `Sources/KittenApp/Resources/coreml/`):
      - `kitten_text_fp32.mlpackage`
      - `kitten_text_fp16.mlpackage`
      - `kitten_text_int8w.mlpackage`
      - `kitten_text_int8wa.mlpackage`
      - `kitten_generator_fp32.mlpackage`
      - `kitten_generator_fp16.mlpackage`
      - `kitten_generator_int8w.mlpackage`
      - `kitten_generator_int8wa.mlpackage`
      (8 files total; previously we shipped 7 per-bucket int8w.)
- [ ] Swift side `TTS.CoreML.swift`: replace `textModels: [Int: MLModel]`
      bucket dict with one `textModel: MLModel?` per variant. Use
      `MLModelConfiguration.functionName = "L_\(bucket)"`. Lazy load
      per (variant, compute) combo.
- [ ] `KittenApp.swift`: add Variant + Compute `@AppStorage` pickers.
      Thread both into `speakOneChunk` via a new `config` knob.
- [ ] Expand metrics log line to include variant + compute tags.
- [ ] Run A/B across the 2×4=8 CoreML configs on macOS + iPhone.
      Populate §11 table.

**Status as of last commit before this INVESTIGATE.md save**: Plan B
scripts exist (`build_multifunction.py`), int8w multifunction files
are built and runtime-tested. fp32 / fp16 / int8wa variants, Swift
loader changes, and UI pickers are NOT YET IMPLEMENTED.

---

**Update (follow-up commit)**: Plan B execution continued. All three
working variants are now bundled and wired end-to-end:

- `scripts/convert_to_coreml.py` now takes `--variant fp32|fp16` and
  writes into `scripts/models/<variant>/`. Converted 5 text × 4 gen
  buckets at both precisions.
- `scripts/quantize_coreml.py` derives `int8w` from `fp32` input via
  `linear_quantize_weights`.
- `scripts/build_multifunction.py` merges 5 text shapes + 4 gen shapes
  per variant into 6 multifunction files.
- `scripts/calibrate_and_quantize.py` exists but blocked — see the
  int8wa row in checklist above.
- `Sources/KittenApp/Resources/coreml/` now contains the 6 shipped
  multifunction mlpackages (fp32 / fp16 / int8w, text + generator).
- `Sources/KittenApp/TTS.CoreML.swift` rewritten: one MLModel per
  `(stage, variant, compute, bucket)` cached lazily. Uses
  `MLModelConfiguration.functionName = "L_\(bucket)"` /
  `"N_\(bucket)"` to pick a function from the multifunction file.
- `Sources/KittenApp/KittenApp.swift` adds **Variant** + **Compute**
  pickers with `@AppStorage`. Disabled when Backend = MLX. Switching
  variant or compute calls `coreMLTTS.unload()` to drop the old
  MLModel instances.
- Log line for CoreML chunks now includes variant/compute tag:
  `CoreML/fp16/All chunk  phonemes=78 L=128 N=256  text ... gen ...`.

**Bundled sizes (`Build/Products/Debug/KittenApp.app/Contents/Resources/`):**

| file                               | size   |
|------------------------------------|--------|
| `kitten_text_fp32.mlmodelc`        | 27 MB  |
| `kitten_text_fp16.mlmodelc`        | 14 MB  |
| `kitten_text_int8w.mlmodelc`       | 7.4 MB |
| `kitten_generator_fp32.mlmodelc`   | 30 MB  |
| `kitten_generator_fp16.mlmodelc`   | 16 MB  |
| `kitten_generator_int8w.mlmodelc`  | 9.1 MB |
| **total CoreML**                   | **104 MB** |
| nano safetensors (MLX)             | 22 MB  |
| voices.safetensors                 | 3 MB   |
| mlx.metallib + mlx-swift bundle    | 7 MB   |
| **Debug Swift dylib**              | 43 MB  |
| **total app bundle**               | **179 MB (Debug)** |

Release build should drop dylib to ~5 MB → ~140 MB. Further cleanup
by pruning losing variants after A/B.

Builds pass on macOS and iOS. Runtime smoke test: app launches,
voice/backend/variant/compute pickers persist via @AppStorage, UI
enabled/disabled logic works. Actual variant × compute grid numbers
not yet measured — that's the next step on device.

---

## 10. Useful existing scripts

All in `scripts/`:

- `torch_kitten.py` — PyTorch port of the full model. TextStage +
  GeneratorStage + KittenTTS (monolithic, uses actual L like ORT).
- `convert_to_coreml.py` — current bucket-based conversion pipeline
  (will be superseded by `convert_dynamic.py` if Plan A succeeds).
- `quantize_coreml.py` — weight-only INT8 via `linear_quantize_weights`.
- `verify_torch_vs_onnx.py` — parity check between our torch port and
  ORT running the same .onnx file.
- `verify_coreml_vs_torch.py` — parity between CoreML .mlpackage output
  and torch port output.
- `compare_three.py` — sanity-check duration predictions across four
  implementations (ONNX / torch mono / torch padded / CoreML).
- `compare_backends.py` — mel-spectrogram diff between two .wav files
  (uses dynamic time-alignment).
- `tail_analysis.py` — diagnose end-of-chunk click (we hit this earlier
  and fixed via tail fade-out in TTS.CoreML.swift).
- `diagnose_lstm_leak.py` — earlier investigation into pad-LSTM drift
  (resolved by choosing nn.LSTM wrapper + smallest-bucket-that-fits).

Reference ONNX file is at `scripts/models/kitten_tts_nano_v0_8.onnx`
(downloaded from HuggingFace, 25 MB INT8).

---

## 11. Running metric comparison (populate during execution)

Template — fill per-row as we ship each variant. Target host: macOS
M-series, iPhone recent. Phrase: "Kitten TTS is now streaming audio
chunks for lower latency." (78 phonemes).

| config                              | macOS RAM | iPhone RAM | macOS RTF | iPhone RTF | audio dur | cold first-speak | note |
|------------------------------------|-----------|------------|-----------|------------|-----------|------------------|------|
| ORT (ref, KittenML impl)            | 265 MB    | —          | ~8×       | —          | —         | instant          | baseline |
| MLX                                 | 1315 MB   | 1315 MB    | 10.6×     | 2.2×       | 6.19 s    | 625 ms           | existing |
| CoreML, pre-investigation, int8w/ANE| 900 MB    | 1669 MB    | 22×       | 1.8× warm  | 6.40 s    | 48 s cold        | ours today |

**Post-Plan-B macOS measurements** (phrase: "Kitten TTS is now streaming
audio chunks for lower latency.", 78 phonemes, L=128 / N=256 buckets).
Baseline RAM: 31 MB before any model. RAM stays ~38–50 MB throughout
because only one variant × compute × bucket pair is loaded at a time.

| variant × compute | load   | warmup | warm text | warm gen | warm total | warm RTF | cold total | cold RTF | quality |
|---|---|---|---|---|---|---|---|---|---|
| **int8w / All** (ANE)   | 254 ms | 1647 ms | 69 ms | 162 ms | **265 ms** | **18.0×** | 2151 ms | 2.2×  | OK |
| fp16  / All (ANE)       | 259 ms | 103 ms  | 41 ms | 125 ms | 568 ms     | 8.4×     | —       | —     | OK |
| fp32  / All (ANE)       | 234 ms | 208 ms  | 50 ms | 134 ms | 668 ms     | 7.2×     | —       | —     | OK |
| fp32  / CPU             | 221 ms | 189 ms  | 52 ms | 158 ms | 660 ms     | 7.3×     | —       | —     | OK |
| fp16  / CPU             | 223 ms |  93 ms  | 40 ms | 120 ms | 512 ms     | 9.4×     | —       | —     | **broken audio** |
| int8w / CPU             | 239 ms | 190 ms  | 52 ms | 159 ms | 680 ms     | 7.0×     | —       | —     | OK |

Peak resident memory during the matrix switch: ~50 MB. **~18× less
than pre-Plan-B** (900 MB → 50 MB) and **~5× less than ORT**.

Key findings so far:

- `int8w / All` is the fastest config (warm RTF 18×) and also the
  smallest on disk (~9 MB generator mlpackage). Matches our goals.
- `fp16 / CPU` produces corrupted audio — probably a coremltools
  issue with fp16 CPU kernels. Not worth chasing if int8w/All wins
  the A/B.
- Three of six (fp32 ANE/CPU, int8w CPU) cluster around RTF 7–9×,
  similar to ORT's baseline. ANE fp16 is faster but identical quality.
- Tail-click fix strengthened: 3 ms cosine fade-in + 20 ms fade-out
  per chunk (was 10 ms fade-out only). Addresses both end-of-utterance
  and between-chunk seams when multi-sentence text splits into
  multiple chunks. Earlier fade may have been too short for int8
  variants whose output has marginally different tail amplitude.
- Very old iPhones (pre-Neural-Engine) run without crashing but
  produce **no audio**. Suspected: audio playback path issue, not a
  model issue. Separate investigation.

**int8wa added as 4th variant (all 8 configs now available for A/B).**
Bundled sizes:

| file                              | size    |
|-----------------------------------|---------|
| `kitten_text_int8wa.mlmodelc`     | 7.9 MB  |
| `kitten_generator_int8wa.mlmodelc`| 9.7 MB  |

Total app bundle on macOS Debug: ~196 MB (was 179 MB). Release build
should still land around 140 MB. int8wa is mostly orthogonal to the
others since its activations actually stay int8 at inference; it's the
only variant that matches ORT's numerical shape exactly.

Matrix for the final A/B run (user to populate from on-device):

|                              | macOS warm RTF | macOS cold | iPhone RTF | iPhone RAM | Audio OK? |
|------------------------------|----------------|------------|-----------|------------|-----------|
| MLX                          | 10.6×          | 625 ms     | 2.2×      | 1315 MB    | yes       |
| CoreML / fp32  / All         | 7.2×           | 668 ms     |           |            | yes       |
| CoreML / fp16  / All         | 8.4×           | 568 ms     |           |            | yes       |
| CoreML / int8w / All         | **18.0×**      | 2151 ms    |           |            | **yes**   |
| CoreML / int8wa/ All         | 18.9×          | **38461 ms** |         |            | **distorted — echoey** |
| CoreML / fp32  / CPU         | 7.3×           | 660 ms     |           |            | yes       |
| CoreML / fp16  / CPU         | 9.4×           | 512 ms     |           |            | **broken** |
| CoreML / int8w / CPU         | 7.0×           | 680 ms     |           |            | yes       |
| CoreML / int8wa/ CPU (=ORT)  | 23.9×          | 2223 ms    |           |            | **distorted — whispering** |

**int8wa quality turned out bad** on both ANE and CPU. Speeds are
competitive (warm 18–24×) but the audio is audibly wrong:

- CPU: "whispering, highly distorted".
- ANE: "echoing like in a cave".
- Cold first-speak on ANE is **38 s** (ANE JIT for a quantized mixed-op
  graph is expensive), making it unusable even if quality improved.

Root-cause hypothesis: our **calibration is synthetic** — the generator
calibration samples are Gaussian noise with crude empirical mean/std
approximations (see `_generator_calibration` in
`scripts/calibrate_and_quantize.py`). Real `prosody_lr`/`text_lr`
tensors produced by the fp32 TextStage + length regulation have
structured distributions (peaks at phoneme boundaries, silence
valleys, different per-channel ranges). Activation ranges computed
from Gaussian noise mismatch the real inference distribution enough
to push quantization outside usable precision.

To fix properly would require: run real sentences through the fp32
TextStage in Python (via `torch_kitten.py`), do Swift-equivalent
length regulation, dump per-sentence `(prosody_lr, text_lr, style)`
tuples matching each bucket N, then re-calibrate. Significant effort
for a variant whose only advantage was "matches ORT exactly" — and
int8w/All already crushes ORT on both RAM and RTF.

**Decision**: `int8wa` stays shipped as a 4th variant for reference
but is **not the winner**. Winner is `CoreML / int8w / All` on both
speed (RTF 18×) and quality. `int8wa` can be revisited if a future
need demands INT8 activations (e.g., a watchOS target where fp16
activation memory matters more).

---

## 12. Context for a fresh session

If you are a new AI session picking this up:

1. Read this whole document.
2. Check `git log -20` to see where we left off.
3. `git status` to see uncommitted work.
4. If Plan A is in flight, look for `scripts/convert_dynamic.py` and
   any uncommitted attempts.
5. Re-open `scripts/torch_kitten.py` — the symbolic-shape audit may
   be partial.
6. Verify the Python env: `scripts/.venv/bin/python -c "import torch,
   coremltools; print(torch.__version__, coremltools.__version__)"`.
   Should be torch 2.11 + coremltools 9.0.

If Plan A has failed, §9 of this document should contain the repro.
Go straight to Plan B (§5).
