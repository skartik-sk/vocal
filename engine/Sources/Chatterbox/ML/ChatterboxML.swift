//
//  ChatterboxML.swift — multilingual Chatterbox (non-turbo) top-level model.
//  Loads mlx-community/chatterbox-4bit and runs text → speech tokens via the
//  multilingual Llama T3. (S3 flow + vocoder port in progress.)
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Multilingual Chatterbox model (chatterbox-4bit).
final class ChatterboxML: Module {
    let config: LlamaT3Config
    @ModuleInfo(key: "t3") var t3: T3ML
    var tokenizer: MTLTokenizer?

    init(config: LlamaT3Config) {
        self.config = config
        _t3.wrappedValue = T3ML(config: config)
        super.init()
    }

    /// Load model weights + tokenizer + conds from a model directory.
    static func fromPretrained(_ modelPath: String) async throws -> ChatterboxML {
        let dir = URL(fileURLWithPath: modelPath)
        let config = LlamaT3Config()   // fixed architecture for chatterbox-4bit

        // Load 4-bit weights (U32 packed + scales/biases).
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))

        // Rename: strip "t3." and map "tfmr.model." -> "tfmr." (drop embed_tokens, lm_head).
        weights = Dictionary(uniqueKeysWithValues: weights.compactMap { (k, v) -> (String, MLXArray)? in
            guard k.hasPrefix("t3.") else { return nil }
            var key = String(k.dropFirst(3))
            key = key.replacingOccurrences(of: "tfmr.model.", with: "tfmr.")
            if key.hasPrefix("tfmr.model.") { key = key.replacingOccurrences(of: "tfmr.model.", with: "tfmr.") }
            if key.contains("embed_tokens") || key.contains("lm_head") { return nil }
            // the Llama backbone holds no rotary_emb params (computed)
            if key.contains("rotary_emb") { return nil }
            return (key, v)
        })

        let model = ChatterboxML(config: config)
        // The checkpoint uses PER-TENSOR group sizes. The quantize filter gets module
        // paths like "t3.speech_head", so key the group-size maps by "t3." + renamed key.
        let groupSizeForPath = ChatterboxLoader.groupSizeForPath(weights).reduce(into: [String: Int]()) {
            $0["t3." + $1.key] = $1.value
        }
        let bitsForPath = ChatterboxLoader.bitsForPath(weights).reduce(into: [String: Int]()) {
            $0["t3." + $1.key] = $1.value
        }
        // Rebuild the weight dict with the full module-tree paths ("t3." prefix) so the
        // packed arrays match applyQuantized's leaf paths.
        var treeWeights = [String: MLXArray]()
        for (k, v) in weights { treeWeights["t3." + k] = v }

        // Dequantize embedding weights (plain Embedding modules hold float weights).
        let embedPaths = ["t3.text_emb", "t3.speech_emb", "t3.text_pos_emb.emb", "t3.speech_pos_emb.emb"]
        for path in embedPaths {
            guard let packed = treeWeights[path + ".weight"],
                  let sc = treeWeights[path + ".scales"] else { continue }
            let bi = treeWeights[path + ".biases"]
            let g = groupSizeForPath[path] ?? 64
            treeWeights[path + ".weight"] = MLX.dequantized(
                packed, scales: sc, biases: bi, groupSize: g, bits: 4, mode: .affine)
            treeWeights.removeValue(forKey: path + ".scales")
            treeWeights.removeValue(forKey: path + ".biases")
        }

        try applyQuantized(model: model, tensors: treeWeights,
                           groupSizeMap: groupSizeForPath, bitsMap: bitsForPath)
        do {
            try model.update(parameters: ModuleParameters.unflattened(treeWeights), verify: [])
        } catch {
            print("[ChatterboxML] ⚠️ update failed: \(error)")
            throw error
        }
        eval(model)

        // Tokenizer (minimal BPE from tokenizer.json)
        let tokenizerData = try Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))
        let bpe = try BPE.load(tokenizerData)
        model.tokenizer = MTLTokenizer(bpe)

        // Baked conds (default voice) — no voice encoder / S3 tokenizer needed.
        if let condsW = try? MLX.loadArrays(url: dir.appendingPathComponent("conds.safetensors")) {
            model.conds = T3MLConds(
                t3SpeakerEmb: condsW["t3.speaker_emb"] ?? MLXArray(0),
                t3EmotionAdv: condsW["t3.emotion_adv"] ?? MLXArray(0),
                t3CondPromptSpeechTokens: condsW["t3.cond_prompt_speech_tokens"] ?? MLXArray(0))
        }

        print("[ChatterboxML] ✅ loaded multilingual (t3 \(config.hiddenLayers)L/\(config.hiddenSize)d)")
        return model
    }

    /// Text → speech tokens (the Hindi gate).
    func speechTokens(text: String, language: String = "hi",
                      temperature: Float = 0.8) -> MLXArray {
        guard let tok = tokenizer else { fatalError("no tokenizer") }
        let ids = tok.tokenize(text: text, languageID: language)
        let textTokens = MLXArray(ids.map { Int32($0) }).reshaped([1, -1])
        let cond = T3MLCond(
            speakerEmb: conds.t3SpeakerEmb,
            emotionAdv: conds.t3EmotionAdv,
            condPromptSpeechTokens: conds.t3CondPromptSpeechTokens)
        return t3.inference(cond: cond, textTokens: textTokens, temperature: temperature)
    }

    // Baked conds (set by loader).
    public var conds: T3MLConds = .empty
}

/// Baked default-voice conditioning from conds.safetensors.
public struct T3MLConds {
    public var t3SpeakerEmb: MLXArray
    public var t3EmotionAdv: MLXArray
    public var t3CondPromptSpeechTokens: MLXArray
    public init(t3SpeakerEmb: MLXArray, t3EmotionAdv: MLXArray,
                t3CondPromptSpeechTokens: MLXArray) {
        self.t3SpeakerEmb = t3SpeakerEmb
        self.t3EmotionAdv = t3EmotionAdv
        self.t3CondPromptSpeechTokens = t3CondPromptSpeechTokens
    }
    public static let empty = T3MLConds(t3SpeakerEmb: MLXArray(0), t3EmotionAdv: MLXArray(0),
                                        t3CondPromptSpeechTokens: MLXArray(0))
}
