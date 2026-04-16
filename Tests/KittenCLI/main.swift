import Foundation
import KittenTTS

@main
struct KittenCLI {
    static func main() async throws {
        setbuf(stdout, nil)
        let text = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1]
            : "Hello from Kittens. This is a self contained text to speech system."
        let voice = CommandLine.arguments.count > 2
            ? CommandLine.arguments[2]
            : "Leo"

        let tts = KittenTTS()
        let config = KittenTTS.Config(speed: 1.0, voiceID: voice)

        print("Synthesizing: \"\(text)\"")
        print("Voice: \(voice)")

        let start = Date()
        var firstByteDate: Date?
        let samples = try await tts.speak(text: text, config: config) { _, _ in
            if firstByteDate == nil { firstByteDate = Date() }
        }
        let end = Date()
        let total = end.timeIntervalSince(start)
        let latency = firstByteDate?.timeIntervalSince(start) ?? total
        let duration = Double(samples.count) / 24000.0

        print("\nResults:")
        print("  First-byte latency: \(String(format: "%.0f", latency * 1000))ms")
        print("  Total time:         \(String(format: "%.0f", total * 1000))ms")
        print("  Audio duration:     \(String(format: "%.2f", duration))s")
        print("  Real-time factor:   \(String(format: "%.1f", duration / total))x")
        print("  Samples:            \(samples.count)")

        // Write WAV
        let output = "output.wav"
        try saveWav(samples: samples, path: output)
        print("  Saved to:           \(output)")
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
