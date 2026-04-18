import SwiftUI
import Combine
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

    func stop() { player.stop() }
}

// MARK: - Small helpers

private extension String {
    func paddedRight(_ width: Int) -> String {
        count >= width ? self : self + String(repeating: " ", count: width - count)
    }
}

// MARK: - Metrics log

enum Backend: String, CaseIterable, Identifiable {
    case mlx = "MLX"
    case coreml = "CoreML"
    var id: String { rawValue }
}

struct LogEntry: Identifiable {
    let id = UUID()
    let time: Date
    let text: String
    let kind: Kind
    enum Kind { case info, metric, warn }
}

@MainActor
final class MetricsLog: ObservableObject {
    @Published var entries: [LogEntry] = []
    @Published var ramMB: Double = 0

    func info(_ s: String) { append(.init(time: Date(), text: s, kind: .info)) }
    func metric(_ s: String) { append(.init(time: Date(), text: s, kind: .metric)) }
    func warn(_ s: String) { append(.init(time: Date(), text: s, kind: .warn)) }

    func updateRAM() { ramMB = KittenMetrics.residentMB() }

    private func append(_ e: LogEntry) {
        entries.append(e)
        if entries.count > 200 { entries.removeFirst(entries.count - 200) }
        ramMB = KittenMetrics.residentMB()
    }
}

// MARK: - Main View

struct KittenTTSView: View {
    @State private var text: String = "Kitten TTS is now streaming audio chunks for lower latency."
    @State private var isGenerating: Bool = false
    @State private var status: String = "Loading model..."
    @State private var modelReady: Bool = false
    @State private var voice: String = "Leo"
    @State private var backend: Backend = .mlx
    @StateObject private var log = MetricsLog()

    private let mlxTTS = KittenTTS()
    private let coreMLTTS = KittenTTSCoreML()
    private let player = AudioPlayer()

    private var voiceOptions: [String] { KittenTTS.voiceDisplayOrder }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Kittens TTS").font(.title).bold()

                TextEditor(text: $text)
                    .frame(minHeight: 90, maxHeight: 140)
                    .padding(6)
                    .overlay(RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.5)))

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Voice").font(.caption).foregroundColor(.secondary)
                        Spacer()
                        Picker("", selection: $voice) {
                            ForEach(voiceOptions, id: \.self) { Text($0).tag($0) }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                    }
                    HStack {
                        Text("Backend").font(.caption).foregroundColor(.secondary)
                        Spacer()
                        Picker("", selection: $backend) {
                            ForEach(Backend.allCases) { Text($0.rawValue).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 200)
                        .disabled(isGenerating)
                    }
                }

                HStack(spacing: 12) {
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

                    Spacer()

                    Text(String(format: "RAM %.0f MB", log.ramMB))
                        .font(.caption.monospaced())
                        .foregroundColor(.secondary)
                }

                Text(status).font(.caption).foregroundColor(.secondary)

                // Metrics log pane.
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        Text("Log").font(.caption.bold())
                        Spacer()
                        Button("Clear") { log.entries.removeAll() }
                            .font(.caption)
                    }
                    .padding(.horizontal, 6)

                    ScrollViewReader { reader in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 2) {
                                ForEach(log.entries) { e in
                                    Text(format(e))
                                        .font(.caption2.monospaced())
                                        .foregroundColor(color(for: e.kind))
                                        .id(e.id)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }
                            .padding(6)
                        }
                        .frame(minHeight: 160, idealHeight: 220)
                        .background(Color.secondary.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .onChange(of: log.entries.count) { _ in
                            if let last = log.entries.last {
                                withAnimation(.linear(duration: 0.08)) {
                                    reader.scrollTo(last.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .task { await preloadModel() }
    }

    private static let timeFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private func format(_ e: LogEntry) -> String {
        "\(Self.timeFmt.string(from: e.time))  \(e.text)"
    }

    private func color(for kind: LogEntry.Kind) -> Color {
        switch kind {
        case .info: return .secondary
        case .metric: return .primary
        case .warn: return .orange
        }
    }

    private func preloadModel() async {
        log.updateRAM()
        log.info("launch  RAM \(Int(log.ramMB)) MB")

        // MLX preload.
        let t0 = Date()
        do {
            try await mlxTTS.preload()
            let ms = Date().timeIntervalSince(t0) * 1000
            log.metric(String(format: "MLX    preload  %5.0f ms", ms))
        } catch {
            log.warn("MLX preload failed: \(error.localizedDescription)")
        }

        // CoreML voice-table preload (models compile lazily on first speak).
        let t1 = Date()
        do {
            try await coreMLTTS.preload()
            let ms = Date().timeIntervalSince(t1) * 1000
            log.metric(String(format: "CoreML voices   %5.0f ms", ms))
        } catch {
            log.warn("CoreML preload failed: \(error.localizedDescription)")
        }

        // Hook metric callbacks.
        coreMLTTS.onBucketCompiled = { [weak log] name, ms in
            let msStr = String(format: "%5.0f", ms)
            Task { @MainActor in
                log?.metric("CoreML compiled \(name.paddedRight(26)) \(msStr) ms")
            }
        }
        coreMLTTS.onChunkMetrics = { [weak log] m in
            let audioS = Double(m.samples) / 24000.0
            let totalMs = m.textStageMs + m.generatorStageMs
            let rtf = audioS / (totalMs / 1000.0)
            Task { @MainActor in
                log?.metric(String(format:
                    "CoreML chunk   phonemes=%d L=%d N=%d  text %.0fms gen %.0fms  audio %.2fs  RTF %.1fx",
                    m.phonemes, m.bucketL, m.bucketN,
                    m.textStageMs, m.generatorStageMs, audioS, rtf))
            }
        }
        mlxTTS.onChunkMetrics = { [weak log] m in
            let audioS = Double(m.samples) / 24000.0
            let rtf = audioS / (m.elapsedMs / 1000.0)
            Task { @MainActor in
                log?.metric(String(format:
                    "MLX    chunk   phonemes=%d                 elapsed %.0fms  audio %.2fs  RTF %.1fx",
                    m.phonemes, m.elapsedMs, audioS, rtf))
            }
        }

        modelReady = true
        status = "Ready"
    }

    private func generateAndStream() {
        isGenerating = true
        let tag = "\(voice) / \(backend.rawValue)"
        status = "Speaking [\(tag)]..."
        player.stop()

        let startTime = Date()
        var firstByteTime: Date?
        let captured = backend
        Task {
            do {
                let cb: (UnsafePointer<Int16>, Int) -> Void = { pointer, count in
                    if firstByteTime == nil { firstByteTime = Date() }
                    let samples = Array(UnsafeBufferPointer(start: pointer, count: count))
                    DispatchQueue.main.async {
                        self.player.playChunk(samples: samples)
                        self.status = "Streaming [\(tag)]..."
                    }
                }
                let totalSamples: Int
                switch captured {
                case .mlx:
                    let cfg = KittenTTS.Config(speed: 1.0, voiceID: voice)
                    let s = try await mlxTTS.speak(text: text, config: cfg, callback: cb)
                    totalSamples = s.count
                case .coreml:
                    let cfg = KittenTTSCoreML.Config(speed: 1.0, voiceID: voice)
                    let s = try await coreMLTTS.speak(text: text, config: cfg, callback: cb)
                    totalSamples = s.count
                }
                let totalMs = Date().timeIntervalSince(startTime) * 1000
                let ttfMs = (firstByteTime ?? Date()).timeIntervalSince(startTime) * 1000
                let audioS = Double(totalSamples) / 24000.0
                let rtf = audioS / (totalMs / 1000.0)
                await MainActor.run {
                    self.isGenerating = false
                    self.status = String(format: "Done [%@] in %.2fs", tag, totalMs / 1000)
                    let backendTag = (captured == .mlx ? "MLX" : "CoreML").paddedRight(6)
                    self.log.metric(String(format:
                        "\(backendTag) SPEAK   TTF %.0fms total %.0fms  audio %.2fs  RTF %.1fx",
                        ttfMs, totalMs, audioS, rtf))
                }
            } catch {
                await MainActor.run {
                    self.isGenerating = false
                    self.status = "Error [\(tag)]: \(error.localizedDescription)"
                    self.log.warn("speak failed: \(error.localizedDescription)")
                }
            }
        }
    }
}
