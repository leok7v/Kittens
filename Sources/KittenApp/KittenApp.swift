import SwiftUI
import Combine
import MLX
import AVFoundation
#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// App entry point. When invoked with `--bench`, we run a headless benchmark
/// and exit without bringing up SwiftUI. Otherwise we hand off to the app.
@main
enum AppEntryPoint {
    static func main() async {
        #if os(macOS)
        let args = CommandLine.arguments
        if args.contains("--bench") {
            await KittenBench.run(args: args)
            exit(0)
        }
        #endif
        KittenApp.main()
    }
}

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
        #if !os(macOS)
        // iOS / iPadOS / visionOS / watchOS: an unconfigured AVAudioSession
        // can default to a category that gets muted by the silent switch
        // (.soloAmbient) or routed nowhere. Force .playback so speech
        // output is audible regardless of ring/silent state.
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio,
                                 options: [.duckOthers])
        try? session.setActive(true, options: [])
        #endif
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

// MARK: - Preset prompts
//
// Bank of sample prompts for quick testing — short demos plus story
// passages from misc/ (dialogues with mixed sentence lengths stress the
// chunker; long paragraphs stress bucketing). Story bodies are bundled
// as .txt files under Resources/prompts/ and loaded on first access.
struct SamplePrompt: Identifiable, Hashable {
    let id: String
    let label: String
    let text: String
}

enum SamplePrompts {
    // (filename-stem, menu label) for stories bundled under Resources/prompts/.
    private static let storyCatalog: [(String, String)] = [
        ("07_baba_yaga",              "Baba Yaga"),
        ("08_shambambukli_creation",  "Shambambukli: Creation"),
        ("09_shambambukli_humans",    "Shambambukli: Humans"),
        ("10_ai_hype",                "AI Hype Unit"),
        ("11_orc_culture",            "High Culture Day"),
        ("12_dark_lord",              "The Dark Lord"),
        ("13_hamish_dougal",          "Hamish and Dougal"),
        ("14_in_the_shire",           "In the Shire"),
        ("15_dragon_contract",        "Dragon Contract"),
        ("16_goblin_park",            "Goblin Park"),
        ("17_leprechaun",             "The Leprechaun"),
        ("18_magic_science",          "Magic is Science"),
        ("19_galactic_exam",          "Galactic Exam"),
    ]

    private static let builtIns: [SamplePrompt] = [
        .init(id: "streaming",
              label: "Streaming demo",
              text: "Kitten TTS is now streaming audio chunks for lower latency."),
        .init(id: "pangram",
              label: "Pangram trio",
              text: """
              The quick brown fox jumps over the lazy dog.

              She sells seashells by the seashore.

              How much wood would a woodchuck chuck if a woodchuck could chuck wood?
              """),
        .init(id: "paragraph",
              label: "Long paragraph",
              text: """
              Speech is one of the most fundamental ways humans communicate with each other, \
              conveying not only the literal meaning of words but also emotion, intent, urgency, \
              humor, and personality through prosody, pitch, and timing. A good text to speech \
              system must capture all of these subtle cues in addition to getting the words \
              themselves right, which is why researchers have spent decades developing better \
              acoustic models, better voice embeddings, and better neural architectures to \
              faithfully reproduce the nuance of the human voice.
              """),
        .init(id: "numbers",
              label: "Numbers & dates",
              text: "In 1969, Apollo 11 landed on the moon. "
                  + "The mission cost about 25.4 billion dollars "
                  + "and brought back 21.5 kilograms of lunar rock."),
    ]

    static let all: [SamplePrompt] = {
        var out = builtIns
        for (stem, label) in storyCatalog {
            guard let url = resourceURL(stem: stem) else { continue }
            guard let body = try? String(contentsOf: url, encoding: .utf8) else { continue }
            out.append(.init(id: stem, label: label,
                             text: body.trimmingCharacters(in: .whitespacesAndNewlines)))
        }
        return out
    }()

    /// Xcode's filesystem-synchronized groups may put prompts/*.txt at the
    /// bundle root or keep them under a `prompts/` subpath. Try both.
    private static func resourceURL(stem: String) -> URL? {
        if let u = Bundle.main.url(forResource: stem, withExtension: "txt") { return u }
        if let base = Bundle.main.resourceURL {
            let u = base.appendingPathComponent("prompts/\(stem).txt")
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        return nil
    }
}

struct KittenTTSView: View {
    @State private var text: String = SamplePrompts.all[0].text
    @AppStorage("prompt") private var promptID: String = SamplePrompts.all[0].id
    @State private var isGenerating: Bool = false
    @State private var status: String = "Loading model..."
    @State private var modelReady: Bool = false
    // Voice + backend + variant + compute persist; speed is session-only.
    @AppStorage("voice")   private var voice: String = "Kiki"
    @AppStorage("backend") private var backend: Backend = .coreml
    @AppStorage("variant") private var variantRaw: String = KittenTTSCoreML.Variant.int8w.rawValue
    @AppStorage("compute") private var computeRaw: String = KittenTTSCoreML.Compute.all.rawValue
    @State private var speed: Float = 1.0
    @State private var speakTask: Task<Void, Never>? = nil
    @StateObject private var log = MetricsLog()

    private var variant: KittenTTSCoreML.Variant {
        get { KittenTTSCoreML.Variant(rawValue: variantRaw) ?? .int8w }
        nonmutating set { variantRaw = newValue.rawValue }
    }
    private var compute: KittenTTSCoreML.Compute {
        get { KittenTTSCoreML.Compute(rawValue: computeRaw) ?? .all }
        nonmutating set { computeRaw = newValue.rawValue }
    }

    private let mlxTTS = KittenTTS()
    private let coreMLTTS = KittenTTSCoreML()
    private let player = AudioPlayer()

    private var voiceOptions: [String] { KittenTTS.voiceDisplayOrder }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
                Text("Kittens TTS").font(.title).bold()

                HStack {
                    Text("Prompt").font(.caption).foregroundColor(.secondary)
                    Spacer()
                    Picker("", selection: $promptID) {
                        ForEach(SamplePrompts.all) { p in
                            Text(p.label).tag(p.id)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                }

                TextEditor(text: $text)
                    .font(.body)
                    .frame(minHeight: 120, maxHeight: .infinity)
                    .padding(6)
                    .overlay(RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.5)))
                    .layoutPriority(1)

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
                    HStack {
                        Text("Variant").font(.caption).foregroundColor(.secondary)
                        Spacer()
                        Picker("", selection: $variantRaw) {
                            ForEach(KittenTTSCoreML.Variant.allCases, id: \.rawValue) { v in
                                Text(v.rawValue).tag(v.rawValue)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 240)
                        .disabled(isGenerating || backend == .mlx)
                    }
                    HStack {
                        Text("Compute").font(.caption).foregroundColor(.secondary)
                        Spacer()
                        Picker("", selection: $computeRaw) {
                            ForEach(KittenTTSCoreML.Compute.allCases, id: \.rawValue) { c in
                                Text(c.rawValue).tag(c.rawValue)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 160)
                        .disabled(isGenerating || backend == .mlx)
                    }
                    HStack {
                        Text("Speed").font(.caption).foregroundColor(.secondary)
                        Slider(value: $speed, in: 0.5...2.0, step: 0.05)
                        Text(String(format: "%.2f×", speed))
                            .font(.caption.monospaced())
                            .frame(width: 52, alignment: .trailing)
                            .foregroundColor(.secondary)
                    }
                }

                HStack(spacing: 12) {
                    // Single toggle button — fixed width so layout doesn't
                    // jump between Speak ↔ Stop states.
                    Button(action: { isGenerating ? stopGeneration() : generateAndStream() }) {
                        Label(isGenerating ? "Stop" : "Speak",
                              systemImage: isGenerating ? "stop.fill" : "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .frame(width: 120)
                    .disabled(!modelReady || (!isGenerating && text.isEmpty))
                    .buttonStyle(.borderedProminent)
                    .tint(isGenerating ? .red : .accentColor)

                    Spacer()

                    Text(String(format: "RAM %.0f MB", log.ramMB))
                        .font(.caption.monospaced())
                        .foregroundColor(.secondary)
                }

                Text(status).font(.caption).foregroundColor(.secondary)

                // Metrics log pane — selectable text, plus a "Copy" button
                // for quick paste into investigation notes / AI sessions.
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        Text("Log").font(.caption.bold())
                        Spacer()
                        Button("Copy") { copyLogToPasteboard() }
                            .font(.caption)
                            .disabled(log.entries.isEmpty)
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
                                        .textSelection(.enabled)
                                }
                            }
                            .padding(6)
                        }
                        .frame(minHeight: 120, maxHeight: 260)
                        .background(Color.secondary.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .onChange(of: log.entries.count) { _, _ in
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
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 560)
        #endif
        .task {
            // @State `text` is initialized before @AppStorage reads the
            // persisted `promptID`, so without this the view always boots
            // showing the builtIns[0] text regardless of saved selection.
            if let p = SamplePrompts.all.first(where: { $0.id == promptID }) {
                text = p.text
            }
            await preloadModel()
        }
        .onChange(of: backend) { _, newValue in
            Task { await switchBackend(to: newValue) }
        }
        .onChange(of: variantRaw) { _, _ in
            // Variant / compute changes invalidate the loaded MLModel set.
            coreMLTTS.unload()
            log.info("variant → \(variantRaw)  (unloaded CoreML models)")
            backgroundWarmUp()
        }
        .onChange(of: computeRaw) { _, _ in
            coreMLTTS.unload()
            log.info("compute → \(computeRaw)  (unloaded CoreML models)")
            backgroundWarmUp()
        }
        .onChange(of: promptID) { _, newID in
            if let p = SamplePrompts.all.first(where: { $0.id == newID }) {
                text = p.text
            }
        }
    }

    private static let timeFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private func format(_ e: LogEntry) -> String {
        "\(Self.timeFmt.string(from: e.time))  \(e.text)"
    }

    private func copyLogToPasteboard() {
        let text = log.entries.map(format).joined(separator: "\n")
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #else
        UIPasteboard.general.string = text
        #endif
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

        // Install metric callbacks once — they survive backend switches.
        coreMLTTS.onBucketLoaded = { [weak log] name, ms in
            let msStr = String(format: "%5.0f", ms)
            Task { @MainActor in
                log?.metric("CoreML loaded   \(name.paddedRight(26)) \(msStr) ms")
            }
        }
        coreMLTTS.onChunkMetrics = { [weak log] m in
            let audioS = Double(m.samples) / 24000.0
            let totalMs = m.textStageMs + m.generatorStageMs
            let rtf = audioS / (totalMs / 1000.0)
            let tag = "\(m.variant.rawValue)/\(m.compute.rawValue)"
            Task { @MainActor in
                log?.metric(String(format:
                    "CoreML/\(tag) chunk  phonemes=%d L=%d N=%d  text %.0fms gen %.0fms  audio %.2fs  RTF %.1fx",
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

        // Only the currently-selected backend is resident at any time.
        await loadBackend(backend)
        modelReady = true
        status = "Ready"

        // Fire-and-forget ANE compile for the default bucket pair so the
        // first Speak doesn't pay that cost in the foreground. The CoreML
        // disk cache persists the compiled .mlmodelc between launches, so
        // this is only slow on first run per (variant × compute × device).
        backgroundWarmUp()
    }

    /// Swap the active backend: drop the old one's memory, preload the new one.
    private func switchBackend(to newBackend: Backend) async {
        modelReady = false
        status = "Switching to \(newBackend.rawValue)..."
        await unloadBackend(newBackend == .mlx ? .coreml : .mlx)
        await loadBackend(newBackend)
        modelReady = true
        status = "Ready"
        backgroundWarmUp()
    }

    private func loadBackend(_ b: Backend) async {
        let tag = (b == .mlx ? "MLX" : "CoreML").paddedRight(6)
        let ramBefore = KittenMetrics.residentMB()
        let t0 = Date()
        do {
            switch b {
            case .mlx:    try await mlxTTS.preload()
            case .coreml: try await coreMLTTS.preload()
            }
            let ms = Date().timeIntervalSince(t0) * 1000
            let ramAfter = KittenMetrics.residentMB()
            log.metric(String(format: "\(tag) preload  %5.0f ms   RAM %.0f → %.0f MB",
                              ms, ramBefore, ramAfter))
        } catch {
            log.warn("\(b.rawValue) preload failed: \(error.localizedDescription)")
        }
    }

    private func unloadBackend(_ b: Backend) async {
        let tag = (b == .mlx ? "MLX" : "CoreML").paddedRight(6)
        let ramBefore = KittenMetrics.residentMB()
        switch b {
        case .mlx:    mlxTTS.unload()
        case .coreml: coreMLTTS.unload()
        }
        // Give autorelease pools a moment to actually drop the pages.
        try? await Task.sleep(nanoseconds: 50_000_000)
        let ramAfter = KittenMetrics.residentMB()
        log.metric(String(format: "\(tag) unload          RAM %.0f → %.0f MB",
                          ramBefore, ramAfter))
    }

    private func stopGeneration() {
        speakTask?.cancel()
        speakTask = nil
        player.stop()
        isGenerating = false
        status = "Stopped"
    }

    /// Kick an off-thread ANE compile for the default bucket pair using
    /// the current variant/compute. Safe to call repeatedly — duplicate
    /// calls for the same (variant, compute, bucket) hit the cache.
    private func backgroundWarmUp() {
        guard backend == .coreml else { return }
        Task.detached(priority: .utility) { [coreMLTTS, variant, compute] in
            await coreMLTTS.warmUpDefault(variant: variant, compute: compute)
        }
    }

    private func generateAndStream() {
        isGenerating = true
        let tag = "\(voice) / \(backend.rawValue)"
        status = "Speaking [\(tag)]..."
        player.stop()

        let startTime = Date()
        var firstByteTime: Date?
        let captured = backend
        // `.utility` keeps inference below the audio I/O thread's
        // priority — since we already generate at RTF > 1×, leaving
        // headroom for AVAudioEngine prevents scrolling / compute from
        // starving audio output on smaller devices (iPhone SE has only
        // 2 performance cores).
        speakTask = Task(priority: .utility) {
            do {
                let cb: (UnsafePointer<Int16>, Int) -> Void = { pointer, count in
                    // If user pressed Stop, abandon further chunks so the
                    // player doesn't get re-scheduled after .stop().
                    if Task.isCancelled { return }
                    if firstByteTime == nil { firstByteTime = Date() }
                    let samples = Array(UnsafeBufferPointer(start: pointer, count: count))
                    // Schedule on the audio thread directly —
                    // AVAudioPlayerNode.scheduleBuffer is thread-safe, and
                    // bouncing every chunk through the main queue stalls
                    // audio during UI scrolling on slower devices.
                    self.player.playChunk(samples: samples)
                    DispatchQueue.main.async {
                        if Task.isCancelled { return }
                        self.status = "Streaming [\(tag)]..."
                    }
                }
                let capturedSpeed = speed
                let totalSamples: Int
                switch captured {
                case .mlx:
                    let cfg = KittenTTS.Config(speed: capturedSpeed, voiceID: voice)
                    let s = try await mlxTTS.speak(text: text, config: cfg, callback: cb)
                    totalSamples = s.count
                case .coreml:
                    let cfg = KittenTTSCoreML.Config(speed: capturedSpeed, voiceID: voice)
                    let s = try await coreMLTTS.speak(
                        text: text, config: cfg,
                        variant: variant, compute: compute,
                        callback: cb)
                    totalSamples = s.count
                }
                if Task.isCancelled { return }
                let totalMs = Date().timeIntervalSince(startTime) * 1000
                let ttfMs = (firstByteTime ?? Date()).timeIntervalSince(startTime) * 1000
                let audioS = Double(totalSamples) / 24000.0
                let rtf = audioS / (totalMs / 1000.0)
                await MainActor.run {
                    self.isGenerating = false
                    self.speakTask = nil
                    self.status = String(format: "Done [%@] in %.2fs", tag, totalMs / 1000)
                    let backendTag = (captured == .mlx ? "MLX" : "CoreML").paddedRight(6)
                    self.log.metric(String(format:
                        "\(backendTag) SPEAK   TTF %.0fms total %.0fms  audio %.2fs  RTF %.1fx",
                        ttfMs, totalMs, audioS, rtf))
                }
            } catch {
                await MainActor.run {
                    self.isGenerating = false
                    self.speakTask = nil
                    self.status = "Error [\(tag)]: \(error.localizedDescription)"
                    self.log.warn("speak failed: \(error.localizedDescription)")
                }
            }
        }
    }
}
