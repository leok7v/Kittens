// v1 is macOS-only — the linked llama.cpp static libs in
// vendors/llama.cpp/build-cpu/ are built for macOS. iOS/xrOS support would
// require building those libs for each platform via xcframework.
#if os(macOS)

import Foundation
import os
import MLX  // shared with the other backends for voices.safetensors loading

/// ggml/llama.cpp-backed alternative to `KittenTTS` and `KittenTTSCoreML`.
/// Same public `speak` API so KittenApp can A/B all three backends.
///
/// Implementation lives in `KittensGGML/kittens-tts.c` (see KittensGGML.h
/// for the C API). The bridging header exposes the C symbols (`kt_create`,
/// `kt_synthesize`, …) directly to Swift.
///
/// v1 caveats:
/// - CPU only. `Compute.metal` requires a custom Metal shader for atan2 in
///   the noise path; not yet wired up.
/// - One graph rebuilt per `kt_synthesize` call. Average speech-rate
///   latency on M-series is ~0.1× realtime per chunk after the first call.
/// - Phonemizer is the shared CEPhonemizer C++ engine — same as the other
///   backends, so input phonemes match.
public final nonisolated class KittenTTSLlamaCpp: @unchecked Sendable {

    public nonisolated struct Config: Sendable {
        public var speed: Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed = speed
            self.voiceID = voiceID
        }
    }

    /// Single backend variant for v1 (CPU only). Kept as an enum so the UI
    /// can present a consistent control across backends.
    public enum Compute: String, Sendable, CaseIterable {
        case cpu = "CPU"
    }

    public typealias SpeakCallback = (UnsafePointer<Int16>, Int) -> Void

    public struct ChunkMetrics: Sendable {
        public let phonemes: Int
        public let frames: Int
        public let compute: Compute
        public let totalMs: Double
        public let samples: Int
    }

    public var onChunkMetrics: ((ChunkMetrics) -> Void)?

    static let audioPerFrame: Int = 600   // 24000 Hz / 80 Hz × 2 (matches CoreML backend)

    private var ctx: OpaquePointer?       // kt_ctx *
    private var voiceEmbeds: [String: [Float]] = [:]   // flattened 400*256
    private let loadLock = OSAllocatedUnfairLock()

    public static var voiceAliases: [String: String] { KittenTTS.voiceAliases }
    public static var voiceDisplayOrder: [String] { KittenTTS.voiceDisplayOrder }

    public init() {
        // Defer GGUF/voices load to preload() — keeps init() cheap.
    }

    deinit {
        if let c = ctx { kt_destroy(c) }
    }

    public func preload() async throws {
        try loadLock.withLock {
            if ctx == nil {
                guard let modelDir = ModelLoader.bundledModelDir() else {
                    throw NSError(domain: "KittenTTSLlamaCpp", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "no bundled model dir"])
                }
                let ggufURL = modelDir.appendingPathComponent("kitten_full.gguf")
                guard FileManager.default.fileExists(atPath: ggufURL.path) else {
                    throw NSError(domain: "KittenTTSLlamaCpp", code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "kitten_full.gguf not found at \(ggufURL.path)"])
                }
                let cgguf = ggufURL.path.cString(using: .utf8)!
                let cbackend = "cpu".cString(using: .utf8)!
                guard let handle = kt_create(cgguf, cbackend) else {
                    throw NSError(domain: "KittenTTSLlamaCpp", code: 3,
                        userInfo: [NSLocalizedDescriptionKey: "kt_create failed"])
                }
                // kt_create returns OpaquePointer? in Swift (kt_ctx is a
                // forward-declared C struct); assign directly.
                ctx = handle
            }
            if voiceEmbeds.isEmpty {
                try loadVoices()
            }
        }
    }

    public func unload() {
        loadLock.withLock {
            if let c = ctx { kt_destroy(c); ctx = nil }
            voiceEmbeds.removeAll()
        }
    }

    public func speak(
        text: String,
        config: Config = Config(),
        compute: Compute = .cpu,
        callback: SpeakCallback? = nil
    ) async throws -> [Float] {
        try await preload()

        let voiceID = KittenTTSLlamaCpp.voiceAliases[config.voiceID] ?? config.voiceID
        guard let voiceRows = voiceEmbeds[voiceID] ?? voiceEmbeds["expr-voice-5-m"] else {
            throw NSError(domain: "KittenTTSLlamaCpp", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "voice '\(voiceID)' not found"])
        }

        let effectiveSpeed = config.speed * (KittenTTS.speedPriors[voiceID] ?? 1.0)
        let normalised = TextPreprocessor.process(text)
        let chunks = TextChunker.chunk(normalised)
        var allAudio: [Float] = []

        // ~120 ms inter-sentence silence (matches CoreML backend).
        let gap = [Float](repeating: 0, count: Int(0.12 * 24000))

        for (idx, chunk) in chunks.enumerated() {
            let phonemes = try Phonemizer.phonemize(chunk)
            let refId = min(chunk.count, 399)
            let style = Array(voiceRows[(refId * 256)..<((refId + 1) * 256)])

            let audio = try synthesizeChunk(phonemes: phonemes, style: style,
                                            speed: effectiveSpeed)
            let emit: [Float] = idx == 0 ? audio : gap + audio
            if let cb = callback {
                let int16 = emit.map { Int16(clamping: Int(($0 * 32767.0).rounded())) }
                int16.withUnsafeBufferPointer { buf in
                    if let base = buf.baseAddress { cb(base, buf.count) }
                }
            }
            allAudio.append(contentsOf: emit)
        }
        return allAudio
    }

    // MARK: - One-chunk pipeline

    private func synthesizeChunk(phonemes: [Int], style: [Float], speed: Float) throws -> [Float] {
        guard let c = ctx else {
            throw NSError(domain: "KittenTTSLlamaCpp", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "context not initialised"])
        }
        precondition(style.count == 256, "style256 must be 256 floats")

        let ids32: [Int32] = phonemes.map { Int32($0) }
        let t0 = Date()
        let audio: kt_audio = ids32.withUnsafeBufferPointer { idsBuf in
            style.withUnsafeBufferPointer { styBuf in
                kt_synthesize(c,
                              idsBuf.baseAddress, Int32(idsBuf.count),
                              styBuf.baseAddress, speed)
            }
        }
        let elapsedMs = Date().timeIntervalSince(t0) * 1000.0

        guard let samples = audio.samples, audio.n_samples > 0 else {
            throw NSError(domain: "KittenTTSLlamaCpp", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "kt_synthesize failed"])
        }
        defer { kt_audio_free(audio) }

        let n = Int(audio.n_samples)
        let buf = UnsafeBufferPointer(start: samples, count: n)
        let out = Array(buf)

        onChunkMetrics?(ChunkMetrics(
            phonemes: phonemes.count,
            frames: n / Self.audioPerFrame,
            compute: .cpu,
            totalMs: elapsedMs,
            samples: n))
        return out
    }

    // MARK: - Voice loading (shared with the other backends)

    private func loadVoices() throws {
        guard let modelDir = ModelLoader.bundledModelDir() else {
            throw NSError(domain: "KittenTTSLlamaCpp", code: 7,
                userInfo: [NSLocalizedDescriptionKey: "no bundled model dir"])
        }
        let url = modelDir.appendingPathComponent("voices.safetensors")
        let raw = try loadArrays(url: url)
        for (name, arr) in raw {
            let flat = arr.asType(.float32).asArray(Float.self)
            voiceEmbeds[name] = flat
        }
    }
}

#endif // os(macOS)
