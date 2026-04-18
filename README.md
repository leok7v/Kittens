# Kittens

On-device text-to-speech for Apple Silicon (macOS + iOS, watchOS planned).
Native-inference Swift app comparing **MLX** and **CoreML** backends for the
KittenTTS nano model, without any `onnxruntime` runtime dependency.

> **Active investigation.** The CoreML backend is mid-rework to match
> `onnxruntime`'s performance and memory envelope. See
> **[`INVESTIGATE.md`](./INVESTIGATE.md)** for the full plan, decisions,
> and restart notes (especially if a new AI session picks up this work).

## Architecture

```
Sources/KittenApp/          SwiftUI app (macOS + iOS)
  TTS.swift                  MLX backend (hand port of kittenForward)
  TTS.CoreML.swift           CoreML backend (consumes .mlpackage files)
  Metrics.swift              RAM + timing helpers
  CEPhonemizer/              C++ IPA phonemizer (Apache 2.0)
  Resources/
    nano/                    safetensors + voices + phonemizer data
    coreml/                  .mlpackage files (compiled to .mlmodelc by Xcode)
KittenML/                    Reference ONNX-Runtime impl for A/B comparison
scripts/                     PyTorch port + coremltools conversion + quantization
Vendor/mlx-swift/            MLX framework (SwiftPM local package)
```

### Pipeline

Text → TextPreprocessor → TextChunker → CEPhonemizer (IPA) → Token IDs →
BERT → Predictor (duration, F0, N) → Length-regulation → HiFi-GAN + iSTFT
→ Audio (24 kHz Int16 PCM)

## Build

Requires Xcode 26+, macOS 15+ SDK.

```bash
open Kittens.xcodeproj
# Build scheme: KittenApp  •  Destination: macOS or iOS
```

## Model

- `KittenML/kitten-tts-nano-0.8-int8` — 15M parameters, 25 MB INT8 ONNX.
- Bundled: original safetensors (MLX), converted `.mlpackage`s (CoreML).

## Voices

Leo · Bella · Jasper · Luna · Bruno · Rosie · Hugo · Kiki
