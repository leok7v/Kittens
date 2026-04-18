import Foundation
import CoreML
import MLX  // used only for loading voices.safetensors; forward pass is all CoreML

/// CoreML-backed alternative to `KittenTTS`. Keeps the same public API so
/// `KittenApp` can A/B the two backends against each other. The text stage
/// runs at fixed L=128 and the generator stage at fixed N=256; longer
/// inputs are chunked upstream via `TextChunker`.
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

    // Bucket sizes that the packaged .mlpackage files were converted for.
    static let textL: Int = 128
    static let generatorN: Int = 256
    static let audioPerFrame: Int = 300  // 24000 Hz / 80 Hz frame rate

    private var textModel: MLModel?
    private var generatorModel: MLModel?
    private var voiceEmbeds: [String: [Float]] = [:]   // flattened 400*256

    // Reused from KittenTTS so callers see the same voice list.
    public static var voiceAliases: [String: String] { KittenTTS.voiceAliases }
    public static var voiceDisplayOrder: [String] { KittenTTS.voiceDisplayOrder }

    public init() {
        // Trigger KittenTTS's metallib install side effect so MLX can later
        // read voices.safetensors without hitting "library not found".
        _ = KittenTTS()
    }

    public func preload() async throws {
        if textModel != nil && generatorModel != nil && !voiceEmbeds.isEmpty { return }
        let config = MLModelConfiguration()
        // .all lets CoreML pick CPU / GPU / ANE per op.
        config.computeUnits = .all

        let textURL = try Self.resourceURL(file: "kitten_text_L128.mlpackage")
        let genURL  = try Self.resourceURL(file: "kitten_generator_N256.mlpackage")
        // MLModel needs a compiled .mlmodelc; compile the .mlpackage lazily.
        let textCompiled = try await MLModel.compileModel(at: textURL)
        let genCompiled = try await MLModel.compileModel(at: genURL)
        textModel = try MLModel(contentsOf: textCompiled, configuration: config)
        generatorModel = try MLModel(contentsOf: genCompiled, configuration: config)
        try loadVoices()
    }

    public func speak(
        text: String,
        config: Config = Config(),
        callback: SpeakCallback? = nil
    ) async throws -> [Float] {
        try await preload()
        guard let textModel, let generatorModel else {
            throw NSError(domain: "KittenTTSCoreML", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "models not loaded"])
        }

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
            let realL = min(phonemes.count, Self.textL)
            let refId = min(chunk.count, 399)
            let style = Array(voiceRows[(refId * 256)..<((refId + 1) * 256)])

            let audio = try speakOneChunk(
                model: textModel, gen: generatorModel,
                phonemes: phonemes, realL: realL,
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

    private func speakOneChunk(model textModel: MLModel, gen generatorModel: MLModel,
                               phonemes: [Int], realL: Int,
                               style: [Float], speed: Float) throws -> [Float] {
        // 1. Build padded input_ids [1, 128] int32.
        let idsArr = try MLMultiArray(shape: [1, NSNumber(value: Self.textL)], dataType: .int32)
        let idsPtr = idsArr.dataPointer.bindMemory(to: Int32.self, capacity: Self.textL)
        for i in 0..<Self.textL {
            idsPtr[i] = i < realL ? Int32(phonemes[i]) : 0
        }
        // 2. Build style [1, 256] float32.
        let styleArr = try MLMultiArray(shape: [1, 256], dataType: .float32)
        let stylePtr = styleArr.dataPointer.bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 { stylePtr[i] = style[i] }

        let textIn = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArr),
            "style": MLFeatureValue(multiArray: styleArr),
        ])
        let textOut = try textModel.prediction(from: textIn)

        // TextStage outputs three MLMultiArrays; identify them by shape.
        var prosodyNCL: MLMultiArray?  // [1, 256, 128]
        var textFeatures: MLMultiArray?  // [1, 128, 128]
        var durSig: MLMultiArray?        // [1, 128, 50]
        for name in textOut.featureNames {
            guard let val = textOut.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = val.shape.map { $0.intValue }
            switch shape {
            case [1, 256, Self.textL]:          prosodyNCL = val
            case [1, 128, Self.textL]:          textFeatures = val
            case [1, Self.textL, 50]:            durSig = val
            default: continue
            }
        }
        guard let prosodyNCL, let textFeatures, let durSig else {
            throw NSError(domain: "KittenTTSCoreML", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "unexpected TextStage outputs"])
        }

        // 3. Compute durations from dur_sig for the real phonemes only.
        let durSigPtr = durSig.dataPointer.bindMemory(to: Float.self, capacity: Self.textL * 50)
        var durations: [Int] = []
        durations.reserveCapacity(realL)
        var totalFrames = 0
        for i in 0..<realL {
            var sum: Float = 0
            let base = i * 50
            for j in 0..<50 { sum += durSigPtr[base + j] }
            let raw = Int((sum / speed).rounded())
            let d = max(1, raw)
            durations.append(d)
            totalFrames += d
        }

        // 4. Cap frames at generatorN, clipping last phonemes if needed.
        if totalFrames > Self.generatorN {
            var overflow = totalFrames - Self.generatorN
            for i in stride(from: durations.count - 1, through: 0, by: -1) {
                let shrink = min(durations[i] - 1, overflow)
                durations[i] -= shrink
                overflow -= shrink
                if overflow <= 0 { break }
            }
            totalFrames = durations.reduce(0, +)
        }

        // 5. Length regulation: write prosody and text_features expanded to [1, C, N=256].
        let prosodyLR = try MLMultiArray(shape: [1, 256, NSNumber(value: Self.generatorN)], dataType: .float32)
        let textLR = try MLMultiArray(shape: [1, 128, NSNumber(value: Self.generatorN)], dataType: .float32)
        let prosodyLRPtr = prosodyLR.dataPointer.bindMemory(to: Float.self, capacity: 256 * Self.generatorN)
        let textLRPtr = textLR.dataPointer.bindMemory(to: Float.self, capacity: 128 * Self.generatorN)
        // Zero pad first.
        for i in 0..<(256 * Self.generatorN) { prosodyLRPtr[i] = 0 }
        for i in 0..<(128 * Self.generatorN) { textLRPtr[i] = 0 }

        let prosodyPtr = prosodyNCL.dataPointer.bindMemory(to: Float.self, capacity: 256 * Self.textL)
        let textPtr = textFeatures.dataPointer.bindMemory(to: Float.self, capacity: 128 * Self.textL)

        var frame = 0
        for i in 0..<realL {
            let d = durations[i]
            for _ in 0..<d {
                if frame >= Self.generatorN { break }
                // Copy prosodyNCL[0, :, i] → prosodyLR[0, :, frame]
                for c in 0..<256 {
                    prosodyLRPtr[c * Self.generatorN + frame] = prosodyPtr[c * Self.textL + i]
                }
                for c in 0..<128 {
                    textLRPtr[c * Self.generatorN + frame] = textPtr[c * Self.textL + i]
                }
                frame += 1
            }
            if frame >= Self.generatorN { break }
        }

        // 6. Run generator.
        let genIn = try MLDictionaryFeatureProvider(dictionary: [
            "prosody_lr": MLFeatureValue(multiArray: prosodyLR),
            "text_lr": MLFeatureValue(multiArray: textLR),
            "style": MLFeatureValue(multiArray: styleArr),
        ])
        let genOut = try generatorModel.prediction(from: genIn)
        guard let wav = genOut.featureNames.compactMap({ genOut.featureValue(for: $0)?.multiArrayValue }).first else {
            throw NSError(domain: "KittenTTSCoreML", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "generator produced no output"])
        }

        // Waveform length for N=256 frames is always 153,600 samples (600 * 256).
        // Real audio is totalFrames * audioPerFrame * 2 (because of F0 upsample).
        let realSamples = totalFrames * Self.audioPerFrame * 2
        let wavCount = wav.count
        let take = min(realSamples, wavCount)
        let wavPtr = wav.dataPointer.bindMemory(to: Float.self, capacity: wavCount)
        return Array(UnsafeBufferPointer(start: wavPtr, count: take))
    }

    // MARK: - Resource / voice loading

    private static func resourceURL(file: String) throws -> URL {
        #if SWIFT_PACKAGE
        if let base = Bundle.module.resourceURL {
            let u = base.appendingPathComponent("coreml").appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        #endif
        throw NSError(domain: "KittenTTSCoreML", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "resource coreml/\(file) not found"])
    }

    private func loadVoices() throws {
        guard let modelDir = ModelLoader.bundledModelDir() else {
            throw NSError(domain: "KittenTTSCoreML", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "no bundled model dir"])
        }
        let url = modelDir.appendingPathComponent("voices.safetensors")
        let raw = try loadArrays(url: url)
        for (name, arr) in raw {
            // Each voice tensor is [400, 256] float32; flatten row-major.
            let flat = arr.asType(.float32).asArray(Float.self)
            voiceEmbeds[name] = flat
        }
    }
}
