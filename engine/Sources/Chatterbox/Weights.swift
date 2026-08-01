//
//  Weights.swift — loading primitives for the native Chatterbox port.
//
//  Handles the two loading concerns the 4-bit checkpoint raises:
//    1. plain safetensors loading (model.safetensors + conds.safetensors);
//    2. the baked default-voice conditioning tensors.
//
//  Quantization layout (verified against the real checkpoint): only **linears and embeddings**
//  are 4-bit (U32 `weight` + F16 `.scales` + `.biases`, all 2-D) — consumed by MLXNN's
//  QuantizedLinear / QuantizedEmbedding at `update()` time. **Convs are full float16** with no
//  scales, so they load as ordinary `Conv1d` weights. No dequantization is needed.
//

import Foundation
import MLX

/// Locates the files inside a Chatterbox-Turbo model directory.
public struct ModelFiles {
    public let dir: URL
    public init(_ path: String) { self.dir = URL(fileURLWithPath: path) }

    public var configURL: URL { dir.appendingPathComponent("config.json") }
    public var weightsURL: URL { dir.appendingPathComponent("model.safetensors") }
    public var condsURL: URL { dir.appendingPathComponent("conds.safetensors") }
}

/// The baked default-voice conditioning (from conds.safetensors). The turbo model ships the
/// entire default voice pre-computed, so the default-voice port never runs the voice encoder.
public struct Conds {
    public let t3SpeakerEmb: MLXArray            // [1, 256]
    public let t3CondPromptSpeechTokens: MLXArray // [1, 375]
    public let genEmbedding: MLXArray            // [1, 192]  (CAMPPlus x-vector)
    public let genPromptFeat: MLXArray           // [1, 500, 80]  (prompt mel)
    public let genPromptToken: MLXArray          // [1, 250]
    public let genPromptTokenLen: MLXArray       // [1]
}

public enum ChatterboxLoader {
    /// Load every tensor from model.safetensors.
    public static func loadWeights(_ files: ModelFiles) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: files.weightsURL)
    }

    /// Load the baked default-voice conds.
    public static func loadConds(_ files: ModelFiles) throws -> Conds {
        let w = try MLX.loadArrays(url: files.condsURL)
        return Conds(
            t3SpeakerEmb: w["t3.speaker_emb"]!,
            t3CondPromptSpeechTokens: w["t3.cond_prompt_speech_tokens"]!,
            genEmbedding: w["gen.embedding"]!,
            genPromptFeat: w["gen.prompt_feat"]!,
            genPromptToken: w["gen.prompt_token"]!,
            genPromptTokenLen: w["gen.prompt_token_len"]!
        )
    }

    /// Paths of quantized tensors (those carrying a `.scales` sibling). These are the
    /// linears + embeddings that MLXNN's `quantize()` will turn into QuantizedLinear /
    /// QuantizedEmbedding so they can consume the packed weight + scales + biases.
    public static func quantizedPaths(_ weights: [String: MLXArray]) -> Set<String> {
        Set(weights.keys
            .filter { $0.hasSuffix(".scales") }
            .map { String($0.dropLast(".scales".count)) })
    }
}
