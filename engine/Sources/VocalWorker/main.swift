//
//  VocalWorker
//
//  Vocal's TTS sidecar. Loads the Qwen3-TTS model ONCE, then loops on stdin:
//  each line is one sentence (sent by the Rust host). Generates speech and
//  plays it through AVAudioEngine. Exits when stdin closes (Rust drops the pipe),
//  which frees all GPU/RAM instantly — the model only lives while there's work.
//

import Foundation
import AVFoundation
import MLX
import Qwen3TTS

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
            print("[VocalWorker] ⚠️ AVAudioEngine failed to start: \(error)")
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
struct VocalWorker {
    static func main() async throws {
        setbuf(stdout, nil) // unbuffered stdout so the Rust host sees logs immediately

        let env = ProcessInfo.processInfo.environment
        let modelPath = env["VOCAL_MODEL_PATH"]
            ?? "/Users/singupallikartik/Developer/fun-projects/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
        let speaker = env["VOCAL_SPEAKER"] ?? "Aiden"
        let language = env["VOCAL_LANGUAGE"] ?? "english"
        // Emotion/style guidance for CustomVoice mode (e.g. "calm, observational").
        // nil = plain delivery (no instruction). Set via VOCAL_INSTRUCT from the Rust host.
        let instruct = env["VOCAL_INSTRUCT"]
        // Sampling temperature. The old hardcoded 0.1 is far too low — it makes the
        // model robotic and more likely to skip/stutter words. 0.8 is the Qwen3-TTS
        // sweet spot; tune via VOCAL_TEMPERATURE from the Rust host.
        let temperature = Float(env["VOCAL_TEMPERATURE"] ?? "0.8") ?? 0.8

        print("[VocalWorker] 🚀 Booting Qwen3-TTS engine...")

        // Audio output (Qwen3-TTS emits 24 kHz mono Float samples).
        let player = AudioPlayer()

        // Load the model exactly once.
        print("[VocalWorker] ⏳ Loading model: \(modelPath)")
        let startLoad = Date()
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("[VocalWorker] ✅ Model loaded in \(String(format: "%.2f", Date().timeIntervalSince(startLoad)))s (speaker=\(speaker), language=\(language), instruct=\(instruct ?? "none"), temp=\(temperature)). Waiting for sentences...")

        // Helper to chunk text by punctuation so it streams faster
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
                print("text:-> \(text) ")
                let startGen = Date()
                
                // Your exact MLX Generation code (Untouched)
                let audio = try await model.generate(
                    text: text,
                    speaker: speaker,
                    instruct: instruct,
                    language: language,
                    temperature: temperature
                )
                eval(audio)
                
                let samples = audio.asArray(Float.self)
                let secs = Double(samples.count) / Double(model.sampleRate)
                print("[VocalWorker] 🗣️ \(String(format: "%.1f", secs))s audio in \(String(format: "%.2f", Date().timeIntervalSince(startGen)))s — playing: \"\(text.prefix(50))\"")
                
                // Use our new non-blocking queue function to instantly jump to the next sentence!
                player.queueAndPlay(samples)

                // Release MLX's cached buffers between sentences to keep memory flat.
                GPU.clearCache()
            }
        }

        // Wait for the final audio sentence to finish playing before closing!
        player.waitUntilFinished()

        print("[VocalWorker] 🛑 stdin closed. Freeing GPU memory and exiting.")
    }
}
