import Foundation
import KittenTTS

@main
struct KittenCLI {
    static func main() async throws {
        setbuf(stdout, nil)
        var args = Array(CommandLine.arguments.dropFirst())
        var backend = "mlx"
        var compare = false
        if let i = args.firstIndex(where: { $0 == "--backend" }), i + 1 < args.count {
            backend = args[i + 1]
            args.removeSubrange(i...(i + 1))
        }
        if let i = args.firstIndex(where: { $0 == "--compare" }) {
            compare = true
            args.remove(at: i)
        }
        let text = args.count > 0
            ? args[0]
            : "Hello from Kittens. This is a self contained text to speech system."
        let voice = args.count > 1 ? args[1] : "Leo"

        if compare {
            try await runComparison(text: text, voice: voice)
            return
        }
        try await runOne(text: text, voice: voice, backend: backend)
    }

    static func runOne(text: String, voice: String, backend: String) async throws {
        print("Synthesizing: \"\(text)\"")
        print("Voice: \(voice)   Backend: \(backend)")
        let (samples, latency, total) = try await synthesize(text: text, voice: voice, backend: backend)
        let duration = Double(samples.count) / 24000.0
        print("\nResults:")
        print("  First-byte latency: \(String(format: "%.0f", latency * 1000))ms")
        print("  Total time:         \(String(format: "%.0f", total * 1000))ms")
        print("  Audio duration:     \(String(format: "%.2f", duration))s")
        print("  Real-time factor:   \(String(format: "%.1f", duration / total))x")
        print("  Samples:            \(samples.count)")
        let output = "output_\(backend).wav"
        try saveWav(samples: samples, path: output)
        print("  Saved to:           \(output)")
    }

    static func runComparison(text: String, voice: String) async throws {
        print("Synthesizing: \"\(text)\"")
        print("Voice: \(voice)")
        // Warm both backends with the ACTUAL text so bucket compile cost is
        // amortized. Running twice gives a proper warm-path measurement.
        _ = try await synthesize(text: text, voice: voice, backend: "mlx")
        _ = try await synthesize(text: text, voice: voice, backend: "coreml")

        let (mlxSamples, mlxLatency, mlxTotal) = try await synthesize(
            text: text, voice: voice, backend: "mlx")
        let (mlSamples, mlLatency, mlTotal) = try await synthesize(
            text: text, voice: voice, backend: "coreml")

        try saveWav(samples: mlxSamples, path: "output_mlx.wav")
        try saveWav(samples: mlSamples,  path: "output_coreml.wav")

        func row(_ name: String, _ samples: [Float], _ first: TimeInterval, _ total: TimeInterval) {
            let dur = Double(samples.count) / 24000.0
            let rtf = dur / total
            print(String(format: "  %-8s  first-byte %6.0fms  total %6.0fms  audio %5.2fs  RTF %4.1fx  samples %6d",
                         (name as NSString).utf8String!, first * 1000, total * 1000, dur, rtf, samples.count))
        }
        print("\nWarm-path timings:")
        row("mlx",    mlxSamples, mlxLatency, mlxTotal)
        row("coreml", mlSamples,  mlLatency,  mlTotal)

        // Numeric diff on prefix (both should be start-aligned).
        let n = min(mlxSamples.count, mlSamples.count)
        if n > 0 {
            var sum: Double = 0; var sqmlx: Double = 0; var sqml: Double = 0; var dot: Double = 0
            for i in 0..<n {
                let a = Double(mlxSamples[i])
                let b = Double(mlSamples[i])
                let d = a - b
                sum += d * d
                sqmlx += a * a
                sqml += b * b
                dot += a * b
            }
            let rms = (sum / Double(n)).squareRoot()
            let cos = dot / (sqmlx.squareRoot() * sqml.squareRoot() + 1e-30)
            print(String(format: "\nWaveform diff on first %d samples:", n))
            print(String(format: "  rms %.4g   cos %.4f   mlx_rms %.4g   coreml_rms %.4g",
                         rms, cos, (sqmlx/Double(n)).squareRoot(), (sqml/Double(n)).squareRoot()))
            print("(Waveforms will not match exactly — sine-gen noise is RNG-dependent. ")
            print(" Pass output_*.wav to scripts/compare_backends.py for mel-spectrogram diff.)")
        }
    }

    static func synthesize(text: String, voice: String, backend: String)
        async throws -> ([Float], TimeInterval, TimeInterval)
    {
        let start = Date()
        var firstByteDate: Date?
        let samples: [Float]
        switch backend {
        case "coreml":
            let tts = KittenTTSCoreML()
            let config = KittenTTSCoreML.Config(speed: 1.0, voiceID: voice)
            samples = try await tts.speak(text: text, config: config) { _, _ in
                if firstByteDate == nil { firstByteDate = Date() }
            }
        default:
            let tts = KittenTTS()
            let config = KittenTTS.Config(speed: 1.0, voiceID: voice)
            samples = try await tts.speak(text: text, config: config) { _, _ in
                if firstByteDate == nil { firstByteDate = Date() }
            }
        }
        let total = Date().timeIntervalSince(start)
        let latency = firstByteDate?.timeIntervalSince(start) ?? total
        return (samples, latency, total)
    }

    static func saveWav(samples: [Float], path: String) throws {
        let url = URL(fileURLWithPath: path)
        let sampleRate: Int32 = 24000
        let channels: Int16 = 1
        let bitsPerSample: Int16 = 16
        var data = Data()
        data.append("RIFF".data(using: .utf8)!)
        let fileSize = Int32(36 + samples.count * 2)
        withUnsafeBytes(of: fileSize) { data.append(contentsOf: $0) }
        data.append("WAVE".data(using: .utf8)!)
        data.append("fmt ".data(using: .utf8)!)
        let fmtSize: Int32 = 16
        withUnsafeBytes(of: fmtSize) { data.append(contentsOf: $0) }
        let formatTag: Int16 = 1
        withUnsafeBytes(of: formatTag) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: channels) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: sampleRate) { data.append(contentsOf: $0) }
        let byteRate = sampleRate * Int32(channels) * Int32(bitsPerSample / 8)
        withUnsafeBytes(of: byteRate) { data.append(contentsOf: $0) }
        let blockAlign = channels * (bitsPerSample / 8)
        withUnsafeBytes(of: blockAlign) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: bitsPerSample) { data.append(contentsOf: $0) }
        data.append("data".data(using: .utf8)!)
        let dataSize = Int32(samples.count * 2)
        withUnsafeBytes(of: dataSize) { data.append(contentsOf: $0) }
        for sample in samples {
            let s = Int16(clamping: Int(round(sample * 32767.0)))
            withUnsafeBytes(of: s) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
    }
}
