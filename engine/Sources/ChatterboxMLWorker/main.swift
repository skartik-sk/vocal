//
//  main.swift — Chatterbox multilingual (Hindi) worker: reads one sentence per
//  line from stdin (sent by the Rust host), synthesizes speech with the native
//  Swift chatterbox-4bit port, and plays it through AVAudioEngine. Exits when
//  stdin closes, freeing the model instantly.
//
//  Env: CHATTERBOX_ML_MODEL (model dir), CHATTERBOX_ML_LANG (hi/en),
//       CHATTERBOX_ML_MAX_TOKENS (cap per sentence).
//

import Foundation
import AVFoundation
import MLX
import Chatterbox

/// Plays 24 kHz mono Float PCM clips sequentially via AVAudioEngine.
final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private var pendingBuffers = 0
    private let queueCondition = NSCondition()

    init(sampleRate: Double = 24000) {
        format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        do { try engine.start() } catch {
            print("[ChatterboxMLWorker] ⚠️ AVAudioEngine failed to start: \(error)")
        }
        node.play()
    }

    func queueAndPlay(_ samples: [Float]) {
        let frames = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return }
        buffer.frameLength = frames
        let dst = buffer.floatChannelData![0]
        for i in 0..<samples.count { dst[i] = samples[i] }

        queueCondition.lock()
        pendingBuffers += 1
        queueCondition.unlock()

        node.scheduleBuffer(buffer) { [weak self] in
            guard let self = self else { return }
            self.queueCondition.lock()
            self.pendingBuffers -= 1
            if self.pendingBuffers == 0 {
                self.queueCondition.signal()
            }
            self.queueCondition.unlock()
        }
    }

    func waitUntilFinished() {
        queueCondition.lock()
        while pendingBuffers > 0 {
            queueCondition.wait()
        }
        queueCondition.unlock()
        Thread.sleep(forTimeInterval: 0.2)
    }
}

@main
struct ChatterboxMLWorker {
    static func main() async throws {
        setbuf(stdout, nil)
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["CHATTERBOX_ML_MODEL"] ?? "/tmp/chatterbox-4bit"
        let lang = env["CHATTERBOX_ML_LANG"] ?? "hi"
        let maxTokens = Int(env["CHATTERBOX_ML_MAX_TOKENS"] ?? "300") ?? 300

        // Cap MLX GPU cache + memory so long sessions stay flat instead of
        // ballooning. The 4-bit model needs ~0.65 GB peak; 1 GB headroom is
        // plenty. relaxed=true lets MLX spill rather than OOM.
        let memMB = Int(env["CHATTERBOX_ML_MEM_MB"] ?? "1024") ?? 1024
        MLX.GPU.set(cacheLimit: memMB * 1024 * 1024)
        MLX.GPU.set(memoryLimit: memMB * 1024 * 1024, relaxed: true)
        print("[ChatterboxMLWorker] memory cap: \(memMB) MB (cached=\(MLX.GPU.cacheMemory), peak=\(MLX.GPU.peakMemory))")

        print("[ChatterboxMLWorker] loading \(modelPath) ...")
        let model = try await ChatterboxML.fromPretrained(modelPath)
        let player = AudioPlayer()
        print("[ChatterboxMLWorker] ready (lang=\(lang)). Waiting for sentences...")

        // Lookahead pipeline: generate the NEXT sentence in the background while
        // the current one plays, so speech is gapless (streaming feel).
        var pending: [(String, [Float])] = []
        // Idle timeout: if no new sentence arrives within this many seconds,
        // exit and free the model (the Tauri host keeps stdin open, so without
        // this the worker lingers in memory after speaking).
        let idleTimeout = Double(env["CHATTERBOX_ML_IDLE_SECS"] ?? "30") ?? 30

        func readLine(timeout: TimeInterval) -> String? {
            // Read stdin with a deadline using a background task + semaphore.
            let sem = DispatchSemaphore(value: 0)
            var result: String? = nil
            DispatchQueue.global().async {
                result = Swift.readLine()
                sem.signal()
            }
            _ = sem.wait(timeout: .now() + timeout)
            return result
        }

        // Pre-fill the first sentence so playback starts immediately.
        if let first = readLine(timeout: idleTimeout) {
            let text = first.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                print("[ChatterboxMLWorker] ⏳ pre-generating first: \"\(text.prefix(40))\"")
                let wav = model.generate(text: text, language: lang, temperature: 0.8,
                                         maxSpeechTokens: maxTokens)
                MLX.GPU.clearCache()
                pending.append((text, wav))
            }
        }

        // Read remaining lines, generating the next sentence during playback.
        while let line = readLine(timeout: idleTimeout) {
            let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { continue }

            // Kick off background generation of this new sentence.
            let genTask = Task {
                let w = model.generate(text: text, language: lang, temperature: 0.8,
                                       maxSpeechTokens: maxTokens)
                MLX.GPU.clearCache()
                return w
            }

            // Play the pending sentence(s) while the next one generates.
            for (t, wav) in pending {
                let secs = Double(wav.count) / 24000.0
                print("[ChatterboxMLWorker] 🗣️ \(String(format: "%.1f", secs))s — \(t.prefix(50))")
                player.queueAndPlay(wav)
            }
            pending.removeAll()

            // Wait for the background generation and queue it next.
            let wav = try await genTask.value
            pending.append((text, wav))
        }

        // Play any final pending sentence, then wait for audio to finish.
        for (t, wav) in pending {
            let secs = Double(wav.count) / 24000.0
            print("[ChatterboxMLWorker] 🗣️ \(String(format: "%.1f", secs))s — \(t.prefix(50))")
            player.queueAndPlay(wav)
        }
        player.waitUntilFinished()
        print("[ChatterboxMLWorker] 🛑 idle timeout (\(Int(idleTimeout))s) or stdin closed. Freeing GPU memory and exiting.")
    }
}
