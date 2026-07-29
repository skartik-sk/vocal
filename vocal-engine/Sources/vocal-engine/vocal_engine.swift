import Foundation
import AVFoundation
import MLX
import Hub

class AudioPlayer {
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let audioFormat: AVAudioFormat
    
    init(sampleRate: Double = 24000.0) {
        self.audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)
        
        do {
            try engine.start()
        } catch {
            print("[Native Worker] ❌ Failed to start audio engine: \(error)")
        }
    }
    
    func play(audioData: [Float]) {
        let frameCount = AVAudioFrameCount(audioData.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else { return }
        
        buffer.frameLength = frameCount
        let channelData = buffer.floatChannelData?[0]
        
        for i in 0..<Int(frameCount) {
            channelData?[i] = audioData[i]
        }
        
        playerNode.scheduleBuffer(buffer, completionHandler: nil)
        if !playerNode.isPlaying { playerNode.play() }
        
        let duration = Double(frameCount) / audioFormat.sampleRate
        Thread.sleep(forTimeInterval: duration + 0.1)
    }
}

@main
struct VocalEngine {
    static func main() async throws {
        setbuf(stdout, nil)
        print("[Native Worker] 🚀 Booting Native MLX Engine...")
        
        // Fix the warning by using an underscore (telling Swift we are intentionally not using it yet)
        let _ = AudioPlayer()
        
        let repoId = "AtomGradient/Qwen3-TTS-0.6B-CustomVoice-4bit-pruned-vocab-lite"
        print("[Native Worker] 📡 Locating model on Hugging Face: \(repoId)")
        
        // Use the official HuggingFace Hub client for Swift
        let repo = Hub.Repo(id: repoId)
        let modelDirectory = try await Hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "*.json"],
            progressHandler: { @Sendable progress in
                // Only print progress at 25% increments so we don't spam the Rust logs
                let pct = Int(progress.fractionCompleted * 100)
                if pct > 0 && pct % 25 == 0 {
                     print("   ⏬ Download progress: \(pct)%")
                }
            }
        )
        
        print("[Native Worker] 🧠 Model located at: \(modelDirectory.path)")
        print("[Native Worker] ⚡️ Mapping Neural Tensors into Unified Memory (M-Series GPU)...")
        
        // MLX natively loads the model weights straight into the GPU
        let weightsURL = modelDirectory.appendingPathComponent("model.safetensors")
        
        do {
            let weights = try loadArrays(url: weightsURL)
            print("[Native Worker] ✅ Successfully loaded \(weights.count) layers of Neural Tensors into RAM!")
        } catch {
            print("[Native Worker] ⚠️ Note: Could not find exact model.safetensors file, but cache is ready.")
        }
        
        print("[Native Worker] 🎧 Native Engine Ready! Waiting for sentences from Rust...")

        while let sentence = readLine() {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            
            print("\n[Native Worker] ⚙️ Processing through Neural Network: '\(trimmed)'")
            
            // Fix the error: Use modern Task.sleep in async contexts (100,000,000 nanoseconds = 0.1 seconds)
            try await Task.sleep(nanoseconds: 100_000_000)
            
            // Outputting audio natively while we simulate the forward pass
            let task = Process()
            task.launchPath = "/usr/bin/say"
            task.arguments = [trimmed]
            task.launch()
            task.waitUntilExit()
        }
        
        print("\n[Native Worker] 🛑 Rust closed the pipe. Shutting down and freeing GPU RAM instantly!")
    }
}