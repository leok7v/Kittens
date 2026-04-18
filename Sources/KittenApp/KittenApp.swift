import SwiftUI
import KittenTTS
import MLX
import AVFoundation

@main
struct KittenApp: App {
    init() {
        #if os(macOS)
        NSApplication.shared.setActivationPolicy(.regular)
        #endif
    }
    var body: some Scene {
        WindowGroup {
            KittenTTSView()
        }
    }
}

// MARK: - Audio Player

class AudioPlayer: NSObject {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let mixer: AVAudioMixerNode
    private let format: AVAudioFormat

    override init() {
        self.mixer = engine.mainMixerNode
        self.format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
        super.init()
        engine.attach(player)
        engine.connect(player, to: mixer, format: format)
        try? engine.start()
    }

    func playChunk(samples: [Int16], sampleRate: Double = 24000) {
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        for i in 0..<samples.count {
            buffer.floatChannelData![0][i] = Float(samples[i]) / 32767.0
        }
        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
        if !player.isPlaying { player.play() }
    }

    func stop() {
        player.stop()
    }
}

// MARK: - Main View

enum Backend: String, CaseIterable, Identifiable {
    case mlx = "MLX"
    case coreml = "CoreML"
    var id: String { rawValue }
}

struct KittenTTSView: View {
    @State private var text: String = "Kitten TTS is now streaming audio chunks for lower latency."
    @State private var isGenerating: Bool = false
    @State private var status: String = "Loading model..."
    @State private var modelReady: Bool = false
    @State private var voice: String = "Leo"
    @State private var backend: Backend = .mlx

    private let mlxTTS = KittenTTS()
    private let coreMLTTS = KittenTTSCoreML()
    private let player = AudioPlayer()

    private var voiceOptions: [String] { KittenTTS.voiceDisplayOrder }

    var body: some View {
        VStack(spacing: 20) {
            Text("KittenTTS")
                .font(.largeTitle)
                .bold()

            TextEditor(text: $text)
                .frame(minHeight: 120)
                .padding(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.5)))

            HStack(spacing: 20) {
                Picker("Voice", selection: $voice) {
                    ForEach(voiceOptions, id: \.self) { v in
                        Text(v).tag(v)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 180)

                Picker("Backend", selection: $backend) {
                    ForEach(Backend.allCases) { b in
                        Text(b.rawValue).tag(b)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 200)
                .disabled(isGenerating)
            }

            HStack(spacing: 15) {
                Button(action: generateAndStream) {
                    if isGenerating {
                        ProgressView().controlSize(.small)
                    } else {
                        Label("Speak", systemImage: "play.fill")
                    }
                }
                .disabled(isGenerating || text.isEmpty || !modelReady)
                .buttonStyle(.borderedProminent)

                Button(action: { player.stop() }) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.bordered)
            }

            Text(status)
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()
        }
        .padding()
        .frame(minWidth: 500, minHeight: 440)
        .task { await preloadModel() }
    }

    private func preloadModel() async {
        do {
            try await mlxTTS.preload()
            // CoreML preload is on demand (compiles .mlpackage the first time).
            modelReady = true
            status = "Ready"
        } catch {
            status = "Failed: \(error.localizedDescription)"
        }
    }

    private func generateAndStream() {
        isGenerating = true
        let tag = "\(voice) / \(backend.rawValue)"
        status = "Speaking [\(tag)]..."
        player.stop()

        let startTime = Date()
        let captured = backend
        Task {
            do {
                let cb: (UnsafePointer<Int16>, Int) -> Void = { pointer, count in
                    let samples = Array(UnsafeBufferPointer(start: pointer, count: count))
                    DispatchQueue.main.async {
                        self.player.playChunk(samples: samples)
                        self.status = "Streaming [\(tag)]..."
                    }
                }
                switch captured {
                case .mlx:
                    let cfg = KittenTTS.Config(speed: 1.0, voiceID: voice)
                    _ = try await mlxTTS.speak(text: text, config: cfg, callback: cb)
                case .coreml:
                    let cfg = KittenTTSCoreML.Config(speed: 1.0, voiceID: voice)
                    _ = try await coreMLTTS.speak(text: text, config: cfg, callback: cb)
                }
                let elapsed = Date().timeIntervalSince(startTime)
                await MainActor.run {
                    self.isGenerating = false
                    self.status = String(format: "Done [%@] in %.2fs", tag, elapsed)
                }
            } catch {
                await MainActor.run {
                    self.isGenerating = false
                    self.status = "Error [\(tag)]: \(error.localizedDescription)"
                }
            }
        }
    }
}
