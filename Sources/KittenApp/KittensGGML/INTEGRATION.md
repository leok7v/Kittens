# KittensGGML — third TTS backend in KittenApp

`KittenTTSLlamaCpp` (in `../TTS.GGML.swift`) is the macOS-only ggml/llama.cpp
backend, sibling to `KittenTTSCoreML` and `KittenTTS` (MLX).

## Build prerequisite (one-time)

Build llama.cpp's static libs for macOS before opening Xcode:

```bash
cd vendors/llama.cpp
cmake -B build-cpu -S . -DGGML_METAL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF
cmake --build build-cpu --config Release -j 8
```

This produces:
```
vendors/llama.cpp/build-cpu/ggml/src/libggml.a
vendors/llama.cpp/build-cpu/ggml/src/libggml-cpu.a
vendors/llama.cpp/build-cpu/ggml/src/libggml-base.a
vendors/llama.cpp/build-cpu/ggml/src/ggml-blas/libggml-blas.a
```

The Xcode project's macOS-conditional linker flags reference these paths.

## What's wired up automatically

- `kittens-tts.c` is auto-included in the KittenApp target's Compile Sources
  via Xcode 15+ filesystem-synchronized groups.
- `Sources/KittenApp/Resources/nano/kitten_full.gguf` (35 MB) is bundled
  alongside `voices.safetensors`.
- `Kittens.xcodeproj/project.pbxproj` already has:
  - `HEADER_SEARCH_PATHS` extended with `$(SRCROOT)/vendors/llama.cpp/ggml/include`
  - `OTHER_LDFLAGS[sdk=macosx*]` listing the four static libs + `-framework Accelerate`
- `kittens-tts.c` is wrapped in `#if TARGET_OS_OSX` so iOS / xrOS targets
  compile it as an empty TU.
- The CLI `main()` is wrapped in `#ifdef KT_BUILD_CLI` so it doesn't collide
  with KittenApp's Swift `@main`. The standalone CLI defines that flag via
  `scripts/build_kittens_tts.sh`.

## What's NOT wired up

- **UI selector.** Whatever enum drives the existing CoreML / MLX backend
  switch needs a `.llamaCpp` (or similar) case + a UI label. Not done here —
  user task.
- **iOS / xrOS support.** Needs an xcframework build of llama.cpp for those
  platforms. Drop the `#if TARGET_OS_OSX` guard once those slices exist.
- **Metal backend.** `Compute` enum has only `.cpu` for v1. Adding Metal
  needs a custom Metal shader for atan2 (the noise path uses it). CPU is
  currently fast enough (~0.1× realtime per chunk on M-series).

## Quick smoke test (Swift API)

```swift
let tts = KittenTTSLlamaCpp()
let audio = try await tts.speak(text: "Hello world.",
                                config: .init(speed: 1.0, voiceID: "Leo"))
// audio is [Float] at 24 kHz mono. Play via AVAudioEngine, write to WAV, etc.
```

## Standalone CLI for debugging

```bash
scripts/build_kittens_tts.sh cpu             # builds tmp/kittens-tts-cpu
python3 scripts/tts_e2e.py --text "Hi"       # text → WAV via the CLI binaries
```
