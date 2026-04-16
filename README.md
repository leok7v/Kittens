# Kittens

Self-contained text-to-speech for Apple Silicon. SwiftUI app + CLI, zero external dependencies.

## Architecture

```
Sources/
  KittenApp/     SwiftUI app (macOS + iOS)
  KittenTTS/     TTS engine library (static)
  CEPhonemizer/  C++ IPA phonemizer (espeak-ng rules)
Tests/
  KittenCLI/     Command-line test harness
Vendor/
  mlx-swift/     MLX framework (vendored, static)
```

### Pipeline

Text → TextPreprocessor (expand $42M, 3rd, 15%) → TextChunker → CEPhonemizer (IPA) → Token IDs → BERT → Prosody/Duration → Decoder → HiFi-GAN + iSTFT → Audio

### Dependencies

**None.** Everything is vendored:

- **MLX** (Apple, MIT) — GPU array framework for Apple Silicon
- **CEPhonemizer** (Apache 2.0) — C++ phonemizer reading espeak-ng rule files
- **espeak-ng data** (GPL v3) — `en_rules` + `en_list` bundled in Resources
- **Model weights** — KittenML/kitten-tts-nano-0.8-int8 (bundled)

## Build

```bash
# CLI test
swift build --product KittenCLI
.build/debug/KittenCLI "Hello world" Leo

# SwiftUI app
swift build --product KittenApp
.build/debug/KittenApp

# Or open in Xcode
open Package.swift
```

## Voices

Leo, Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki

## Performance (Apple Silicon, nano model)

- First-byte latency: ~500ms
- Real-time factor: 4-10x
- Memory: ~40-60MB
