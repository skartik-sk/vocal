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
        // Local dir of the Qwen3-TTS 0.6B CustomVoice 4-bit model. The Rust host always
        // sets VOCAL_MODEL_PATH from vocal.config; this is only a dev fallback — point it
        // at your own downloaded model dir.
        let modelPath = env["VOCAL_MODEL_PATH"]
            ?? "~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
        let speaker = env["VOCAL_SPEAKER"] ?? "Aiden"
        let language = env["VOCAL_LANGUAGE"] ?? "english"
        // Emotion/style guidance for CustomVoice mode (e.g. "calm, observational").
        // nil = plain delivery (no instruction). Rust sends "" when none is set, so
        // treat empty/whitespace as "no instruct".
        let rawInstruct = env["VOCAL_INSTRUCT"]
        let instruct = (rawInstruct?.trimmingCharacters(in: .whitespaces).isEmpty ?? true) ? nil : rawInstruct
        // Sampling temperature. 0.8 is the Qwen3-TTS sweet spot; tune via VOCAL_TEMPERATURE.
        let temperature = Float(env["VOCAL_TEMPERATURE"] ?? "0.8") ?? 0.8

        // Playback mode. Default = whole-clip (generate the full clause, then play): it is
        // smoother and far lower memory, because at <1× realtime the extra decode work that
        // streaming interleaves onto the GPU starves the playback buffer and balloons memory.
        // Set VOCAL_STREAM=1 to opt into chunked streaming (faster first-audio, but choppy
        // and heavier until generation runs faster than realtime).
        let useStream = (env["VOCAL_STREAM"] ?? "0") != "0"

        print("[VocalWorker] 🚀 Booting Qwen3-TTS engine... (mode: \(useStream ? "streaming" : "whole-clip"))")

        // Audio output (Qwen3-TTS emits 24 kHz mono Float samples).
        let player = AudioPlayer()

        // Load the model exactly once.
        print("[VocalWorker] ⏳ Loading model: \(modelPath)")
        let startLoad = Date()
        let model = try await Qwen3TTSModel.fromPretrained(modelPath)
        print("[VocalWorker] ✅ Model loaded in \(String(format: "%.2f", Date().timeIntervalSince(startLoad)))s (speaker=\(speaker), language=\(language), instruct=\(instruct ?? "none"), temp=\(temperature)). Waiting for sentences...")

        // Helper to chunk text by punctuation so it streams faster.
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

                if useStream {
                    // Streaming: yield ~0.6s audio chunks as codec tokens are produced,
                    // feeding the non-blocking player so playback overlaps generation.
                    let stream = model.generateAudioStream(
                        text: text, speaker: speaker, instruct: instruct,
                        language: language, temperature: temperature)
                    var firstChunkAt: TimeInterval = 0
                    var chunkCount = 0
                    for try await event in stream {
                        if case .chunk(let samples) = event {
                            if firstChunkAt == 0 { firstChunkAt = Date().timeIntervalSince(startGen) }
                            chunkCount += 1
                            player.queueAndPlay(samples)
                        }
                    }
                    print("[VocalWorker] 🌊 \(chunkCount) chunks, first in \(String(format: "%.2f", firstChunkAt))s — playing: \"\(text.prefix(50))\"")
                } else {
                    // Whole-clip fallback (proven path).
                    let audio = try await model.generate(
                        text: text, speaker: speaker, instruct: instruct,
                        language: language, temperature: temperature)
                    eval(audio)
                    let samples = audio.asArray(Float.self)
                    let secs = Double(samples.count) / Double(model.sampleRate)
                    print("[VocalWorker] 🗣️ \(String(format: "%.1f", secs))s audio in \(String(format: "%.2f", Date().timeIntervalSince(startGen)))s — playing: \"\(text.prefix(50))\"")
                    player.queueAndPlay(samples)
                }

                // Release MLX's cached buffers between sentences to keep memory flat.
                GPU.clearCache()
            }
        }

        // Wait for the final audio sentence to finish playing before closing!
        player.waitUntilFinished()

        print("[VocalWorker] 🛑 stdin closed. Freeing GPU memory and exiting.")
    }
}
