import Foundation
@preconcurrency import CoreML
import os
import MLX  // used only for loading voices.safetensors; forward pass is all CoreML

/// CoreML-backed alternative to `KittenTTS`. Keeps the same public API so
/// `KittenApp` can A/B the two backends against each other.
///
/// **Multifunction** `.mlpackage` layout (as of Plan B in INVESTIGATE.md):
/// one file per (stage × variant); each file contains a set of pre-compiled
/// functions (`L_16`, `L_32`, …, `N_1024`) that share weights on disk and
/// in memory. Swift picks the right function via
/// `MLModelConfiguration.functionName`.
///
/// Dimensions the backend exposes for A/B:
/// - `Variant`: precision. `fp32` / `fp16` / `int8w`. (`int8wa` blocked by
///   a coremltools bug on int32 inputs — see INVESTIGATE.md.)
/// - `Compute`: `.all` (let CoreML pick ANE/GPU/CPU) or `.cpuOnly`.
public final nonisolated class KittenTTSCoreML: @unchecked Sendable {

    public nonisolated struct Config: Sendable {
        public var speed: Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed = speed
            self.voiceID = voiceID
        }
    }

    public enum Variant: String, Sendable, CaseIterable {
        case fp32
        case fp16
        case int8w
    }

    public enum Compute: String, Sendable, CaseIterable {
        case all     = "All"
        case cpuOnly = "CPU"
    }

    public typealias SpeakCallback = (UnsafePointer<Int16>, Int) -> Void

    /// Metrics emitted by the backend for UI / CLI display.
    public struct ChunkMetrics: Sendable {
        public let phonemes: Int
        public let bucketL: Int
        public let bucketN: Int
        public let variant: Variant
        public let compute: Compute
        public let textStageMs: Double
        public let generatorStageMs: Double
        public let samples: Int
    }

    public var onBucketLoaded: ((_ name: String, _ elapsedMs: Double) -> Void)?
    public var onChunkMetrics: ((ChunkMetrics) -> Void)?

    // Shipped bucket (function) sizes. Must match the function names inside
    // the multifunction .mlpackages. Keep sorted ascending.
    static let textBuckets: [Int] = [16, 32, 64, 128, 400]
    static let generatorBuckets: [Int] = [128, 256, 512, 1024]
    static let audioPerFrame: Int = 300  // 24000 Hz / 80 Hz frame rate × 2 (F0 upsample)

    /// Cache key: different variant / compute / bucket combos are distinct
    /// MLModel instances because each binds its own MLModelConfiguration.
    private struct Key: Hashable {
        let stage: String        // "text" | "generator"
        let variant: Variant
        let compute: Compute
        let bucket: Int
    }

    private var models: [Key: MLModel] = [:]
    private var voiceEmbeds: [String: [Float]] = [:]   // flattened 400*256
    private let loadLock = OSAllocatedUnfairLock()

    public static var voiceAliases: [String: String] { KittenTTS.voiceAliases }
    public static var voiceDisplayOrder: [String] { KittenTTS.voiceDisplayOrder }

    public init() {
        _ = KittenTTS()  // triggers metallib install for MLX-shared voices load
    }

    public func preload() async throws {
        if voiceEmbeds.isEmpty { try loadVoices() }
    }

    /// Drop all loaded MLModel instances and anything they compiled.
    public func unload() {
        loadLock.withLock { models.removeAll() }
    }

    public func speak(
        text: String,
        config: Config = Config(),
        variant: Variant = .int8w,
        compute: Compute = .all,
        callback: SpeakCallback? = nil
    ) async throws -> [Float] {
        try await preload()

        let voiceID = KittenTTSCoreML.voiceAliases[config.voiceID] ?? config.voiceID
        guard let voiceRows = voiceEmbeds[voiceID] ?? voiceEmbeds["expr-voice-5-m"] else {
            throw NSError(domain: "KittenTTSCoreML", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "voice '\(voiceID)' not found"])
        }

        let effectiveSpeed = config.speed * (KittenTTS.speedPriors[voiceID] ?? 1.0)
        let normalised = TextPreprocessor.process(text)
        let chunks = TextChunker.chunk(normalised)
        var allAudio: [Float] = []

        for chunk in chunks {
            let phonemes = try Phonemizer.phonemize(chunk)
            let refId = min(chunk.count, 399)
            let style = Array(voiceRows[(refId * 256)..<((refId + 1) * 256)])

            let audio = try await speakOneChunk(
                phonemes: phonemes, style: style, speed: effectiveSpeed,
                variant: variant, compute: compute)
            if let cb = callback {
                let int16 = audio.map { Int16(clamping: Int(($0 * 32767.0).rounded())) }
                int16.withUnsafeBufferPointer { buf in
                    if let base = buf.baseAddress { cb(base, buf.count) }
                }
            }
            allAudio.append(contentsOf: audio)
        }
        return allAudio
    }

    // MARK: - Pipeline for a single chunk

    private func speakOneChunk(
        phonemes: [Int], style: [Float], speed: Float,
        variant: Variant, compute: Compute
    ) async throws -> [Float] {
        let realL = phonemes.count
        let L = Self.textBuckets.first(where: { $0 >= realL }) ?? Self.textBuckets.last!
        let clippedL = min(realL, L)
        let textModel = try await model(stage: "text", variant: variant,
                                        compute: compute, bucket: L)
        let tTextStart = Date()

        let idsArr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        let idsPtr = idsArr.dataPointer.bindMemory(to: Int32.self, capacity: L)
        let maskArr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float32)
        let maskPtr = maskArr.dataPointer.bindMemory(to: Float.self, capacity: L)
        for i in 0..<L {
            idsPtr[i] = i < clippedL ? Int32(phonemes[i]) : 0
            maskPtr[i] = i < clippedL ? 1.0 : 0.0
        }
        let styleArr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let stylePtr = styleArr.dataPointer.bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 { stylePtr[i] = style[i] }

        let textIn = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: idsArr),
            "style":          MLFeatureValue(multiArray: styleArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let textOut = try await textModel.prediction(from: textIn)

        var prosodyNCL: MLMultiArray?
        var textFeatures: MLMultiArray?
        var durSig: MLMultiArray?
        for name in textOut.featureNames {
            guard let val = textOut.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = val.shape.map { $0.intValue }
            switch shape {
            case [1, 256, L]: prosodyNCL = val
            case [1, 128, L]: textFeatures = val
            case [1, L, 50]:  durSig = val
            default: continue
            }
        }
        guard let prosodyNCL, let textFeatures, let durSig else {
            throw NSError(domain: "KittenTTSCoreML", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "unexpected TextStage outputs"])
        }

        let durSigFlat = try Self.copyToFloat32(durSig)
        let prosodyFlat = try Self.copyToFloat32(prosodyNCL)
        let textFlat = try Self.copyToFloat32(textFeatures)
        let textStageMs = Date().timeIntervalSince(tTextStart) * 1000.0

        var durations: [Int] = []
        durations.reserveCapacity(clippedL)
        var totalFrames = 0
        for i in 0..<clippedL {
            var sum: Float = 0
            let base = i * 50
            for j in 0..<50 { sum += durSigFlat[base + j] }
            let raw = Int((sum / speed).rounded())
            let d = max(1, raw)
            durations.append(d)
            totalFrames += d
        }

        let nBucket = Self.generatorBuckets.first(where: { $0 >= totalFrames })
            ?? Self.generatorBuckets.last!
        if totalFrames > nBucket {
            var overflow = totalFrames - nBucket
            for i in stride(from: durations.count - 1, through: 0, by: -1) {
                let shrink = min(durations[i] - 1, overflow)
                durations[i] -= shrink
                overflow -= shrink
                if overflow <= 0 { break }
            }
            totalFrames = durations.reduce(0, +)
        }
        let generatorModel = try await model(stage: "generator", variant: variant,
                                             compute: compute, bucket: nBucket)
        let tGenStart = Date()

        let prosodyLR = try MLMultiArray(shape: [1, 256, NSNumber(value: nBucket)], dataType: .float32)
        let textLR = try MLMultiArray(shape: [1, 128, NSNumber(value: nBucket)], dataType: .float32)
        let prosodyLRPtr = prosodyLR.dataPointer.bindMemory(to: Float.self, capacity: 256 * nBucket)
        let textLRPtr = textLR.dataPointer.bindMemory(to: Float.self, capacity: 128 * nBucket)
        for i in 0..<(256 * nBucket) { prosodyLRPtr[i] = 0 }
        for i in 0..<(128 * nBucket) { textLRPtr[i] = 0 }

        var frame = 0
        for i in 0..<durations.count {
            let d = durations[i]
            for _ in 0..<d {
                if frame >= nBucket { break }
                for c in 0..<256 {
                    prosodyLRPtr[c * nBucket + frame] = prosodyFlat[c * L + i]
                }
                for c in 0..<128 {
                    textLRPtr[c * nBucket + frame] = textFlat[c * L + i]
                }
                frame += 1
            }
            if frame >= nBucket { break }
        }

        let genIn = try MLDictionaryFeatureProvider(dictionary: [
            "prosody_lr": MLFeatureValue(multiArray: prosodyLR),
            "text_lr":    MLFeatureValue(multiArray: textLR),
            "style":      MLFeatureValue(multiArray: styleArr),
        ])
        let genOut = try await generatorModel.prediction(from: genIn)
        guard let wav = genOut.featureNames.compactMap({ genOut.featureValue(for: $0)?.multiArrayValue }).first else {
            throw NSError(domain: "KittenTTSCoreML", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "generator produced no output"])
        }

        let wavFlat = try Self.copyToFloat32(wav)
        let realSamples = totalFrames * Self.audioPerFrame * 2
        let take = min(realSamples, wavFlat.count)
        var output = Array(wavFlat.prefix(take))
        Self.applyTailFade(&output, fadeSamples: 240)
        let generatorStageMs = Date().timeIntervalSince(tGenStart) * 1000.0
        onChunkMetrics?(ChunkMetrics(
            phonemes: realL, bucketL: L, bucketN: nBucket,
            variant: variant, compute: compute,
            textStageMs: textStageMs, generatorStageMs: generatorStageMs,
            samples: output.count))
        return output
    }

    /// Cosine fade the last `fadeSamples` of `samples` from 1× down to 0.
    private static func applyTailFade(_ samples: inout [Float], fadeSamples: Int) {
        let n = min(fadeSamples, samples.count)
        guard n > 0 else { return }
        let start = samples.count - n
        for i in 0..<n {
            let t = Float(i) / Float(n - 1 > 0 ? n - 1 : 1)
            let gain = 0.5 + 0.5 * cos(.pi * t)
            samples[start + i] *= gain
        }
    }

    /// Copy an MLMultiArray into a packed row-major Float32 array, regardless
    /// of its native dtype (fp16 vs fp32) or padded strides.
    private static func copyToFloat32(_ a: MLMultiArray) throws -> [Float] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let packed = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: packed)
        var indices = Array(repeating: 0, count: shape.count)
        let rank = shape.count

        func walkFp16() {
            let src = a.dataPointer.assumingMemoryBound(to: UInt16.self)
            for dst in 0..<packed {
                var srcOff = 0
                for k in 0..<rank { srcOff += indices[k] * strides[k] }
                out[dst] = Float(Float16(bitPattern: src[srcOff]))
                for k in stride(from: rank - 1, through: 0, by: -1) {
                    indices[k] += 1
                    if indices[k] < shape[k] { break }
                    indices[k] = 0
                }
            }
        }
        func walkFp32() {
            let src = a.dataPointer.assumingMemoryBound(to: Float.self)
            for dst in 0..<packed {
                var srcOff = 0
                for k in 0..<rank { srcOff += indices[k] * strides[k] }
                out[dst] = src[srcOff]
                for k in stride(from: rank - 1, through: 0, by: -1) {
                    indices[k] += 1
                    if indices[k] < shape[k] { break }
                    indices[k] = 0
                }
            }
        }
        switch a.dataType {
        case .float16: walkFp16()
        case .float32: walkFp32()
        default:
            throw NSError(domain: "KittenTTSCoreML", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "unsupported output dtype \(a.dataType.rawValue)"])
        }
        return out
    }

    // MARK: - Lazy model loading (one MLModel per variant × compute × bucket)

    private func model(stage: String, variant: Variant, compute: Compute,
                       bucket: Int) async throws -> MLModel {
        let key = Key(stage: stage, variant: variant, compute: compute, bucket: bucket)
        if let cached = loadLock.withLock({ models[key] }) { return cached }

        let packageName = "kitten_\(stage)_\(variant.rawValue)"   // e.g. kitten_text_fp16
        let axis = stage == "text" ? "L" : "N"
        let functionName = "\(axis)_\(bucket)"

        let cfg = MLModelConfiguration()
        cfg.computeUnits = {
            switch compute {
            case .all:     return .all
            case .cpuOnly: return .cpuOnly
            }
        }()
        cfg.functionName = functionName

        guard let found = Self.resourceURL(name: packageName) else {
            throw NSError(domain: "KittenTTSCoreML", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "bundle has no \(packageName).mlmodelc or \(packageName).mlpackage"])
        }
        let tLoadStart = Date()
        let loaded: MLModel
        if found.isCompiled {
            loaded = try MLModel(contentsOf: found.url, configuration: cfg)
        } else {
            let compiled = try await Self.compiledModelURL(packageURL: found.url, name: packageName)
            loaded = try MLModel(contentsOf: compiled, configuration: cfg)
        }
        let loadMs = Date().timeIntervalSince(tLoadStart) * 1000.0

        let tWarmStart = Date()
        try await warmup(model: loaded, stage: stage, bucket: bucket)
        let warmMs = Date().timeIntervalSince(tWarmStart) * 1000.0

        let label = "\(packageName) [\(functionName)/\(compute.rawValue)]  load \(Int(loadMs))ms + warmup"
        onBucketLoaded?(label, warmMs)

        loadLock.withLock { models[key] = loaded }
        return loaded
    }

    private func warmup(model: MLModel, stage: String, bucket: Int) async throws {
        let input: MLDictionaryFeatureProvider
        if stage == "text" {
            let L = bucket
            let ids = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
            let style = try MLMultiArray(shape: [1, 256], dataType: .float32)
            let mask = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float32)
            memset(ids.dataPointer,   0, L * MemoryLayout<Int32>.size)
            memset(style.dataPointer, 0, 256 * MemoryLayout<Float>.size)
            memset(mask.dataPointer,  0, L * MemoryLayout<Float>.size)
            input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids":      MLFeatureValue(multiArray: ids),
                "style":          MLFeatureValue(multiArray: style),
                "attention_mask": MLFeatureValue(multiArray: mask),
            ])
        } else {
            let N = bucket
            let prosody = try MLMultiArray(shape: [1, 256, NSNumber(value: N)], dataType: .float32)
            let text    = try MLMultiArray(shape: [1, 128, NSNumber(value: N)], dataType: .float32)
            let style   = try MLMultiArray(shape: [1, 256], dataType: .float32)
            memset(prosody.dataPointer, 0, 256 * N * MemoryLayout<Float>.size)
            memset(text.dataPointer,    0, 128 * N * MemoryLayout<Float>.size)
            memset(style.dataPointer,   0, 256 * MemoryLayout<Float>.size)
            input = try MLDictionaryFeatureProvider(dictionary: [
                "prosody_lr": MLFeatureValue(multiArray: prosody),
                "text_lr":    MLFeatureValue(multiArray: text),
                "style":      MLFeatureValue(multiArray: style),
            ])
        }
        _ = try await model.prediction(from: input)
    }

    private static func resourceURL(name: String) -> (url: URL, isCompiled: Bool)? {
        for ext in ["mlmodelc", "mlpackage"] {
            if let u = Bundle.main.url(forResource: name, withExtension: ext) {
                return (u, ext == "mlmodelc")
            }
        }
        if let base = Bundle.main.resourceURL {
            for ext in ["mlmodelc", "mlpackage"] {
                let u = base.appendingPathComponent("coreml/\(name).\(ext)")
                if FileManager.default.fileExists(atPath: u.path) { return (u, ext == "mlmodelc") }
            }
        }
        return nil
    }

    /// Return a compiled `.mlmodelc` URL, reusing a cached copy if its mtime
    /// matches the bundled `.mlpackage`'s Manifest. First launch pays the
    /// compile cost once per package.
    private static func compiledModelURL(packageURL: URL, name: String) async throws -> URL {
        let fm = FileManager.default
        let cacheRoot = try fm.url(for: .cachesDirectory, in: .userDomainMask,
                                   appropriateFor: nil, create: true)
            .appendingPathComponent("KittenTTS", isDirectory: true)
        try? fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let manifestURL = packageURL.appendingPathComponent("Manifest.json")
        let pkgMtime = (try? fm.attributesOfItem(atPath: manifestURL.path)[.modificationDate] as? Date)
            ?? Date.distantPast
        let stamp = Int(pkgMtime.timeIntervalSince1970)
        let compiledURL = cacheRoot.appendingPathComponent("\(name)_\(stamp).mlmodelc", isDirectory: true)
        if fm.fileExists(atPath: compiledURL.path) { return compiledURL }
        if let entries = try? fm.contentsOfDirectory(at: cacheRoot, includingPropertiesForKeys: nil) {
            for url in entries where url.lastPathComponent.hasPrefix("\(name)_") {
                try? fm.removeItem(at: url)
            }
        }
        let freshlyCompiled = try await MLModel.compileModel(at: packageURL)
        try? fm.removeItem(at: compiledURL)
        try fm.moveItem(at: freshlyCompiled, to: compiledURL)
        return compiledURL
    }

    private func loadVoices() throws {
        guard let modelDir = ModelLoader.bundledModelDir() else {
            throw NSError(domain: "KittenTTSCoreML", code: 6,
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
