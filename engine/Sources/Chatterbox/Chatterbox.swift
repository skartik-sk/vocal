//
//  Chatterbox  — native Swift/MLX port of Chatterbox-Turbo (default voice)
//
//  Pure-Swift reimplementation (mirrors how swift-qwen3-tts ports Qwen3-TTS), so this
//  backend has ZERO Python dependency. Ported from the Python reference in
//  `mlx_audio.tts.models.chatterbox_turbo`.
//
//  MILESTONE STATUS:
//    M0 (now): skeleton only — `fromPretrained` is a no-op, `generate` returns a short
//              tone so the ChatterboxWorker plumbing can be exercised end to end.
//    M1+: real weight loading, T3 (GPT-2), S3 (Conformer encoder + mean-flow CFM), and the
//         HiFTNet vocoder fill in below.
//

import Foundation

/// Top-level native Chatterbox-Turbo model.
public final class ChatterboxTurbo {
    public static let sampleRate = 24000

    public init() {}

    /// Load model weights + the baked `conds.safetensors` default voice from a model directory.
    ///
    /// M0 stub: no-op (returns an empty model). Implemented in M1 once the weight-loading
    /// recipe (MLX.loadArrays → detect `.scales` → quantize → update) and the Chatterbox
    /// config structs are in place.
    public static func fromPretrained(_ modelPath: String) async throws -> ChatterboxTurbo {
        print("[Chatterbox] (M0 skeleton) would load 4-bit model from: \(modelPath)")
        return ChatterboxTurbo()
    }

    /// Text → 24 kHz mono Float samples.
    ///
    /// M0 stub: returns a 0.4 s tone so the worker can be heard and the plumbing verified
    /// before any neural code lands. M5 replaces this with T3 → S3 → HiFTNet.
    public func generate(text: String) -> [Float] {
        print("[Chatterbox] (M0 skeleton) generating stub tone for: \"\(text.prefix(60))\"")
        return ChatterboxTurbo.stubTone()
    }

    /// A short 440 Hz tone, used by the M0 stub so playback wiring is audible.
    public static func stubTone() -> [Float] {
        let sr = sampleRate
        let n = Int(0.4 * Double(sr))
        return (0..<n).map { i in Float(sin(2.0 * .pi * 440.0 * Double(i) / Double(sr)) * 0.2) }
    }
}
