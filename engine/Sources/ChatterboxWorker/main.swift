//
//  ChatterboxWorker
//
//  Vocal's NATIVE Chatterbox sidecar (pure Swift/MLX — no Python). Loads the 4-bit
//  Chatterbox-Turbo model ONCE, then loops on stdin: each line is one sentence (sent by
//  the Rust host). Generates speech and plays it through AVAudioEngine. Exits when stdin
//  closes (Rust drops the pipe), freeing all GPU/RAM instantly.
//
//  Speaks the SAME stdin protocol as the Qwen3 `VocalWorker`, so `config.rs` can route to
//  it via `backend = native_chatterbox`.
//

import Foundation
import AVFoundation
import Chatterbox

/// Plays 24 kHz mono Float PCM clips sequentially via AVAudioEngine.
final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat

    // Thread-safe counters to track background audio chunks
    private var pendingBuffers = 0
    private let queueCondition = NSCondition()

    init(sampleRate: Double = 24000) {
        format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        do { try engine.start() } catch {
            print("[ChatterboxWorker] ⚠️ AVAudioEngine failed to start: \(error)")
        }
        node.play()
    }

    /// Schedules a clip in the background and returns INSTANTLY.
    func queueAndPlay(_ samples: [Float]) {
        let frames = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return }
        buffer.frameLength = frames
        let dst = buffer.floatChannelData![0]
        for i in 0..<samples.count { dst[i] = samples[i] }

        // Safely register that a new clip is playing
        queueCondition.lock()
        pendingBuffers += 1
        queueCondition.unlock()

        // Pass it to the background audio thread
        node.scheduleBuffer(buffer) { [weak self] in
            guard let self = self else { return }
            self.queueCondition.lock()
            self.pendingBuffers -= 1
            if self.pendingBuffers == 0 {
                self.queueCondition.signal() // Signal if the app is waiting to shut down
            }
            self.queueCondition.unlock()
        }
    }

    /// Blocks only at the very end of the app to ensure the final sentence finishes.
    func waitUntilFinished() {
        queueCondition.lock()
        while pendingBuffers > 0 {
            queueCondition.wait()
        }
        queueCondition.unlock()
        // Give audio hardware a fraction of a second to push the final soundwaves
        Thread.sleep(forTimeInterval: 0.2)
    }
}

@main
struct ChatterboxWorker {
    static func main() async throws {
        setbuf(stdout, nil) // unbuffered stdout so the Rust host sees logs immediately

        let env = ProcessInfo.processInfo.environment
        let modelPath = env["CHATTERBOX_MODEL_PATH"]
            ?? "/Users/singupallikartik/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-4bit/snapshots/c63817725071d7b5269c7b558772d6e8cbf59cec"

        print("[ChatterboxWorker] 🚀 Booting native Chatterbox engine...")

        // Audio output (Chatterbox emits 24 kHz mono Float samples).
        let player = AudioPlayer()

        // Load the model exactly once.
        print("[ChatterboxWorker] ⏳ Loading model: \(modelPath)")
        let startLoad = Date()
        let model = try await ChatterboxTurbo.fromPretrained(modelPath)
        print("[ChatterboxWorker] ✅ Model loaded in \(String(format: "%.2f", Date().timeIntervalSince(startLoad)))s. Waiting for sentences...")

        // Helper to chunk text by punctuation so long inputs play sooner.
        func chunkText(_ text: String) -> [String] {
            var chunks = [String]()
            var currentChunk = ""
            for char in text {
                currentChunk.append(char)
                if char == "." || char == "!" || char == "?" || char == ";" || char == "," {
                    if currentChunk.count > 15 { // avoid tiny chunks
                        chunks.append(currentChunk.trimmingCharacters(in: .whitespaces))
                        currentChunk = ""
                    }
                }
            }
            if !currentChunk.trimmingCharacters(in: .whitespaces).isEmpty {
                chunks.append(currentChunk.trimmingCharacters(in: .whitespaces))
            }
            return chunks.isEmpty ? [text] : chunks
        }

        // Read sentences from Rust, one per line.
        while let line = readLine() {
            let fullText = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if fullText.isEmpty { continue }

            let clauses = chunkText(fullText)
            for text in clauses {
                let startGen = Date()
                let samples = model.generate(text: text)
                let secs = Double(samples.count) / Double(ChatterboxTurbo.sampleRate)
                print("[ChatterboxWorker] 🗣️ \(String(format: "%.1f", secs))s audio in \(String(format: "%.2f", Date().timeIntervalSince(startGen)))s — playing: \"\(text.prefix(50))\"")
                player.queueAndPlay(samples)
            }
        }

        // Wait for the final audio sentence to finish playing before closing!
        player.waitUntilFinished()

        print("[ChatterboxWorker] 🛑 stdin closed. Freeing GPU memory and exiting.")
    }
}
