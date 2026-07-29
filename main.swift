import Foundation
import MLX

func main() {
    print("[Native Worker] Loading MLX Model into Unified Memory...")
    
    // Load the MLX version of your Qwen3 model natively
    let model = try! loadModel(id: "AtomGradient/Qwen3-TTS-0.6B-CustomVoice-4bit-pruned-vocab-lite")
    
    // Read from Rust via standard input (just like we did with Python)
    while let sentence = readLine() {
        print("Generating native audio for: \(sentence)")
        let audioData = model.generate(text: sentence)
        playAudio(data: audioData)
    }
}
main()
