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

struct KittenTTSView: View {
    @State private var text: String = "Kitten TTS is now streaming audio chunks for lower latency."
    @State private var isGenerating: Bool = false
    @State private var status: String = "Loading model..."
    @State private var modelReady: Bool = false
    @State private var voice: String = "Leo"

    private let tts = KittenTTS()
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

            Picker("Voice", selection: $voice) {
                ForEach(voiceOptions, id: \.self) { v in
                    Text(v).tag(v)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 180)

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
            try await tts.preload()
            modelReady = true
            status = "Ready"
        } catch {
            status = "Failed: \(error.localizedDescription)"
        }
    }

    private func generateAndStream() {
        isGenerating = true
        status = "Speaking [\(voice)]..."
        player.stop()

        let config = KittenTTS.Config(speed: 1.0, voiceID: voice)

        Task {
            do {
                _ = try await tts.speak(text: text, config: config) { pointer, count in
                    let samples = Array(UnsafeBufferPointer(start: pointer, count: count))
                    DispatchQueue.main.async {
                        self.player.playChunk(samples: samples)
                        self.status = "Streaming [\(voice)]..."
                    }
                }
                await MainActor.run {
                    self.isGenerating = false
                    self.status = "Done [\(voice)]"
                }
            } catch {
                await MainActor.run {
                    self.isGenerating = false
                    self.status = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
}
