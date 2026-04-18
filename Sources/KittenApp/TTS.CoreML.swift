import Foundation
import CoreML
import MLX  // used only for loading voices.safetensors; forward pass is all CoreML

/// CoreML-backed alternative to `KittenTTS`. Keeps the same public API so
/// `KittenApp` can A/B the two backends against each other.
///
/// coremltools 9 flex shapes convert but fail at runtime, so we ship a
/// bucket set (several fixed-L TextStage packages, several fixed-N
/// GeneratorStage packages) and pick the smallest bucket that fits each
/// chunk. Buckets are compiled on first use (≈5s cold cost per bucket)
/// and cached on disk in Caches/KittenTTS/.
public final class KittenTTSCoreML: @unchecked Sendable {

    public struct Config {
        public var speed: Float
        public var voiceID: String
        public init(speed: Float = 1.0, voiceID: String = "Leo") {
            self.speed = speed
            self.voiceID = voiceID
        }
    }

    public typealias SpeakCallback = (UnsafePointer<Int16>, Int) -> Void

    /// Metrics emitted by the backend for UI / CLI display.
    public struct ChunkMetrics: Sendable {
        public let phonemes: Int
        public let bucketL: Int
        public let bucketN: Int
        public let textStageMs: Double
        public let generatorStageMs: Double
        public let samples: Int
    }

    /// Called after a bucket is ready to use — for bundle-precompiled
    /// `.mlmodelc`s this is just the MLModel load time; for `.mlpackage`s
    /// it includes Apple's on-device compile cost.
    public var onBucketLoaded: ((_ name: String, _ elapsedMs: Double) -> Void)?
    /// Called once per speak() chunk, after audio is produced.
    public var onChunkMetrics: ((ChunkMetrics) -> Void)?

    // Shipped bucket sizes. Keep sorted ascending.
    static let textBuckets: [Int] = [16, 32, 64, 128]
    static let generatorBuckets: [Int] = [128, 256, 512]
    static let audioPerFrame: Int = 300  // 24000 Hz / 80 Hz frame rate × 2 (F0 upsample)

    private var textModels: [Int: MLModel] = [:]
    private var generatorModels: [Int: MLModel] = [:]
    private var voiceEmbeds: [String: [Float]] = [:]   // flattened 400*256
    private let loadLock = NSLock()

    public static var voiceAliases: [String: String] { KittenTTS.voiceAliases }
    public static var voiceDisplayOrder: [String] { KittenTTS.voiceDisplayOrder }

    public init() {
        // Trigger KittenTTS's metallib install side effect so MLX can later
        // read voices.safetensors without hitting "library not found".
        _ = KittenTTS()
    }

    /// Load voice table, all bucket models, and run a dummy inference through
    /// each model so ANE's first-use JIT compile happens now (while the UI is
    /// still showing "Loading…") instead of on the user's first speak.
    /// Takes a few seconds per backend switch but makes every speak fast.
    public func preload() async throws {
        if voiceEmbeds.isEmpty { try loadVoices() }
        for L in Self.textBuckets {
            _ = try await textBucket(L)
        }
        for N in Self.generatorBuckets {
            _ = try await generatorBucket(N)
        }
    }

    /// Release all loaded MLModel buckets (voice table stays — it's 400 KB).
    /// Call this when switching to another backend to free RAM.
    public func unload() {
        loadLock.lock()
        textModels.removeAll()
        generatorModels.removeAll()
        loadLock.unlock()
    }

    public func speak(
        text: String,
        config: Config = Config(),
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
                phonemes: phonemes,
                style: style, speed: effectiveSpeed)
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

    private func speakOneChunk(phonemes: [Int], style: [Float], speed: Float) async throws -> [Float] {
        // Pick the smallest text bucket that fits, clipping to max if longer.
        let realL = phonemes.count
        let L = Self.textBuckets.first(where: { $0 >= realL }) ?? Self.textBuckets.last!
        let clippedL = min(realL, L)
        let textModel = try await textBucket(L)
        let tTextStart = Date()

        // 1. Build padded input_ids [1, L] int32 + attention_mask [1, L] float32.
        let idsArr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        let idsPtr = idsArr.dataPointer.bindMemory(to: Int32.self, capacity: L)
        let maskArr = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float32)
        let maskPtr = maskArr.dataPointer.bindMemory(to: Float.self, capacity: L)
        for i in 0..<L {
            idsPtr[i] = i < clippedL ? Int32(phonemes[i]) : 0
            maskPtr[i] = i < clippedL ? 1.0 : 0.0
        }
        // 2. Build style [1, 256] float32.
        let styleArr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let stylePtr = styleArr.dataPointer.bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 { stylePtr[i] = style[i] }

        let textIn = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArr),
            "style": MLFeatureValue(multiArray: styleArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let textOut = try await textModel.prediction(from: textIn)

        var prosodyNCL: MLMultiArray?   // [1, 256, L]
        var textFeatures: MLMultiArray? // [1, 128, L]
        var durSig: MLMultiArray?       // [1, L, 50]
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

        // MLMultiArrays from a MIL program arrive as float16 with padded
        // strides on Apple Silicon (e.g. (1,128,50) stored as (8192,64,1) —
        // last dim padded to 64 for SIMD alignment). Reading the raw
        // dataPointer as Float32 gives garbage. Copy into packed fp32 arrays.
        let durSigFlat = try Self.copyToFloat32(durSig)
        let prosodyFlat = try Self.copyToFloat32(prosodyNCL)
        let textFlat = try Self.copyToFloat32(textFeatures)
        let textStageMs = Date().timeIntervalSince(tTextStart) * 1000.0

        // 3. Compute durations from dur_sig for the real phonemes only.
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

        // 4. Pick smallest generator bucket that fits; clip durations if needed.
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
        let generatorModel = try await generatorBucket(nBucket)
        let tGenStart = Date()

        // 5. Length regulation into [1, C, N].
        let prosodyLR = try MLMultiArray(shape: [1, 256, NSNumber(value: nBucket)], dataType: .float32)
        let textLR = try MLMultiArray(shape: [1, 128, NSNumber(value: nBucket)], dataType: .float32)
        let prosodyLRPtr = prosodyLR.dataPointer.bindMemory(to: Float.self, capacity: 256 * nBucket)
        let textLRPtr = textLR.dataPointer.bindMemory(to: Float.self, capacity: 128 * nBucket)
        // Zero pad first.
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

        // 6. Run generator.
        let genIn = try MLDictionaryFeatureProvider(dictionary: [
            "prosody_lr": MLFeatureValue(multiArray: prosodyLR),
            "text_lr": MLFeatureValue(multiArray: textLR),
            "style": MLFeatureValue(multiArray: styleArr),
        ])
        let genOut = try await generatorModel.prediction(from: genIn)
        guard let wav = genOut.featureNames.compactMap({ genOut.featureValue(for: $0)?.multiArrayValue }).first else {
            throw NSError(domain: "KittenTTSCoreML", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "generator produced no output"])
        }

        // Output length is 600 * nBucket samples; real audio is 600 * totalFrames.
        let wavFlat = try Self.copyToFloat32(wav)
        let realSamples = totalFrames * Self.audioPerFrame * 2
        let take = min(realSamples, wavFlat.count)
        let generatorStageMs = Date().timeIntervalSince(tGenStart) * 1000.0
        onChunkMetrics?(ChunkMetrics(
            phonemes: realL, bucketL: L, bucketN: nBucket,
            textStageMs: textStageMs, generatorStageMs: generatorStageMs,
            samples: take))
        return Array(wavFlat.prefix(take))
    }

    /// Copy an MLMultiArray into a packed row-major Float32 array, regardless
    /// of its native dtype (fp16 vs fp32) or padded strides.
    private static func copyToFloat32(_ a: MLMultiArray) throws -> [Float] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let packed = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: packed)

        // Walk every packed index, translate to the (possibly padded) src offset.
        var indices = Array(repeating: 0, count: shape.count)
        let rank = shape.count

        func copyFromFp16() {
            let src = a.dataPointer.assumingMemoryBound(to: UInt16.self)
            for dst in 0..<packed {
                var srcOff = 0
                for k in 0..<rank { srcOff += indices[k] * strides[k] }
                // Decode IEEE 754 half via Float16 bit pattern.
                let bits = src[srcOff]
                out[dst] = Float(Float16(bitPattern: bits))
                // Advance the index vector (row-major).
                for k in stride(from: rank - 1, through: 0, by: -1) {
                    indices[k] += 1
                    if indices[k] < shape[k] { break }
                    indices[k] = 0
                }
            }
        }

        func copyFromFp32() {
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
        case .float16: copyFromFp16()
        case .float32: copyFromFp32()
        default:
            throw NSError(domain: "KittenTTSCoreML", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "unsupported output dtype \(a.dataType.rawValue)"])
        }
        return out
    }

    // MARK: - Lazy bucket loaders

    private func textBucket(_ L: Int) async throws -> MLModel {
        if let cached = textModels[L] { return cached }
        let m = try await compileAndLoad(name: "kitten_text_L\(L)")
        loadLock.lock(); textModels[L] = m; loadLock.unlock()
        return m
    }

    private func generatorBucket(_ N: Int) async throws -> MLModel {
        if let cached = generatorModels[N] { return cached }
        let m = try await compileAndLoad(name: "kitten_generator_N\(N)")
        loadLock.lock(); generatorModels[N] = m; loadLock.unlock()
        return m
    }

    private func compileAndLoad(name: String) async throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let found = Self.resourceURL(name: name) else {
            throw NSError(domain: "KittenTTSCoreML", code: 5,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "bundle has no \(name).mlmodelc or \(name).mlpackage"])
        }
        let loadStart = Date()
        let model: MLModel
        var displayName = name
        if found.isCompiled {
            model = try MLModel(contentsOf: found.url, configuration: cfg)
            displayName += " (precompiled)"
        } else {
            // Fallback: compile .mlpackage on first use, cache compiled bundle.
            let compiledURL = try await Self.compiledModelURL(packageURL: found.url, name: name)
            model = try MLModel(contentsOf: compiledURL, configuration: cfg)
        }
        let loadMs = Date().timeIntervalSince(loadStart) * 1000.0

        // Run one dummy inference to trigger ANE's on-first-use JIT compile
        // now, rather than on the user's first speak. Adds ~0.5–2s per bucket
        // to preload but moves the cost off the user's critical path.
        let warmStart = Date()
        try await warmup(model: model, name: name)
        let warmMs = Date().timeIntervalSince(warmStart) * 1000.0

        onBucketLoaded?("\(displayName) load \(Int(loadMs))ms + warmup",
                         warmMs)
        return model
    }

    /// Feed a zero-filled input through the model once, forcing ANE to do
    /// its JIT compile while we're still in "loading" UX. We discard the
    /// output.
    private func warmup(model: MLModel, name: String) async throws {
        let input: MLDictionaryFeatureProvider
        if let L = Self.bucketSize(name: name, prefix: "kitten_text_L") {
            let ids = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
            let style = try MLMultiArray(shape: [1, 256], dataType: .float32)
            let mask = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float32)
            // MLMultiArray buffers aren't necessarily zero-initialised — memset.
            Self.zeroFill(ids, count: L, stride: MemoryLayout<Int32>.size)
            Self.zeroFill(style, count: 256, stride: MemoryLayout<Float>.size)
            Self.zeroFill(mask, count: L, stride: MemoryLayout<Float>.size)
            input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids":       MLFeatureValue(multiArray: ids),
                "style":           MLFeatureValue(multiArray: style),
                "attention_mask":  MLFeatureValue(multiArray: mask),
            ])
        } else if let N = Self.bucketSize(name: name, prefix: "kitten_generator_N") {
            let prosody = try MLMultiArray(shape: [1, 256, NSNumber(value: N)], dataType: .float32)
            let text = try MLMultiArray(shape: [1, 128, NSNumber(value: N)], dataType: .float32)
            let style = try MLMultiArray(shape: [1, 256], dataType: .float32)
            Self.zeroFill(prosody, count: 256 * N, stride: MemoryLayout<Float>.size)
            Self.zeroFill(text, count: 128 * N, stride: MemoryLayout<Float>.size)
            Self.zeroFill(style, count: 256, stride: MemoryLayout<Float>.size)
            input = try MLDictionaryFeatureProvider(dictionary: [
                "prosody_lr": MLFeatureValue(multiArray: prosody),
                "text_lr":    MLFeatureValue(multiArray: text),
                "style":      MLFeatureValue(multiArray: style),
            ])
        } else {
            return  // unknown model name, skip warmup
        }
        _ = try await model.prediction(from: input)
    }

    private static func bucketSize(name: String, prefix: String) -> Int? {
        guard name.hasPrefix(prefix) else { return nil }
        return Int(name.dropFirst(prefix.count))
    }

    private static func zeroFill(_ a: MLMultiArray, count: Int, stride: Int) {
        memset(a.dataPointer, 0, count * stride)
    }

    /// Return a compiled `.mlmodelc` URL, reusing a cached copy if its mtime
    /// matches the bundled `.mlpackage`'s manifest. First launch pays ~5s per
    /// bucket to compile; subsequent launches are near-instant.
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

        // Evict stale compiled copies for this bucket name.
        if let entries = try? fm.contentsOfDirectory(at: cacheRoot, includingPropertiesForKeys: nil) {
            for url in entries where url.lastPathComponent.hasPrefix("\(name)_") {
                try? fm.removeItem(at: url)
            }
        }
        let freshlyCompiled = try await MLModel.compileModel(at: packageURL)
        // `compileModel` returns a URL in the OS tmp dir; move it into the cache.
        try? fm.removeItem(at: compiledURL)
        try fm.moveItem(at: freshlyCompiled, to: compiledURL)
        return compiledURL
    }

    /// Locate a CoreML resource (precompiled .mlmodelc preferred, .mlpackage
    /// fallback). Checks folder-reference paths (`coreml/<name>.<ext>`) and
    /// flat layouts. Returns whichever exists.
    private static func resourceURL(name: String) -> (url: URL, isCompiled: Bool)? {
        for ext in ["mlmodelc", "mlpackage"] {
            if let u = Bundle.main.url(forResource: name, withExtension: ext) {
                return (u, ext == "mlmodelc")
            }
        }
        // Xcode may preserve our "coreml/" subfolder as a folder reference.
        if let base = Bundle.main.resourceURL {
            for ext in ["mlmodelc", "mlpackage"] {
                let u = base.appendingPathComponent("coreml/\(name).\(ext)")
                if FileManager.default.fileExists(atPath: u.path) {
                    return (u, ext == "mlmodelc")
                }
            }
        }
        return nil
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
