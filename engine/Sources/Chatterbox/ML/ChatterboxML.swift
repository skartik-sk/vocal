//
//  ChatterboxML.swift — multilingual Chatterbox (non-turbo) top-level model.
//  Loads mlx-community/chatterbox-4bit and runs text → speech tokens via the
//  multilingual Llama T3, then S3 flow (10-step CFM) → mel → HiFT vocoder → wav.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Multilingual Chatterbox model (chatterbox-4bit).
final class ChatterboxML: Module {
    let config: LlamaT3Config
    @ModuleInfo(key: "t3") var t3: T3ML
    @ModuleInfo(key: "flow") var flow: MLFlow
    @ModuleInfo(key: "mel2wav") var mel2wav: HiFTGenerator
    var tokenizer: MTLTokenizer?

    init(config: LlamaT3Config) {
        self.config = config
        _t3.wrappedValue = T3ML(config: config)
        _flow.wrappedValue = MLFlow()
        _mel2wav.wrappedValue = HiFTGenerator(samplingRate: 24000)
        super.init()
    }

    /// Load model weights + tokenizer + conds from a model directory.
    static func fromPretrained(_ modelPath: String) async throws -> ChatterboxML {
        let dir = URL(fileURLWithPath: modelPath)
        let config = LlamaT3Config()   // fixed architecture for chatterbox-4bit
        let allWeights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))

        // ---- Split + rename by component ----
        // t3: strip "t3.", map "tfmr.model." -> "tfmr.", drop embed_tokens/lm_head/rotary_emb,
        // then re-add "t3." so keys match the module-tree paths.
        var weights = Dictionary(uniqueKeysWithValues: allWeights.compactMap { (k, v) -> (String, MLXArray)? in
            guard k.hasPrefix("t3.") else { return nil }
            var key = String(k.dropFirst(3))
            key = key.replacingOccurrences(of: "tfmr.model.", with: "tfmr.")
            if key.contains("embed_tokens") || key.contains("lm_head") || key.contains("rotary_emb") {
                return nil
            }
            return ("t3." + key, v)
        })
        // s3gen.flow.* -> flow.* (keep the "flow" prefix to match MLFlow tree),
        // renaming decoder blocks down_blocks_N -> down_blocks.N (array form) and
        // pos_bias_u/v -> .weight (raw params stored as Embedding.weight).
        for (k, v) in allWeights where k.hasPrefix("s3gen.flow.") {
            var key = "flow." + String(k.dropFirst("s3gen.flow.".count))
            key = key.replacingOccurrences(of: "down_blocks_", with: "down_blocks.")
            key = key.replacingOccurrences(of: "mid_blocks_", with: "mid_blocks.")
            key = key.replacingOccurrences(of: "up_blocks_", with: "up_blocks.")
            key = key.replacingOccurrences(of: "encoders_", with: "encoders.")
            key = key.replacingOccurrences(of: "up_encoders_", with: "up_encoders.")
            key = key.replacingOccurrences(of: "transformer_", with: "transformer_blocks.")
            if key.hasSuffix(".pos_bias_u") || key.hasSuffix(".pos_bias_v") {
                weights[key + ".weight"] = v
            } else if key.hasPrefix("flow.decoder.estimator.final_proj.") {
                // Conv1dPT wraps the conv in `.conv`.
                weights[key.replacingOccurrences(of: "final_proj.", with: "final_proj.conv.")] = v
            } else {
                weights[key] = v
            }
        }
        // s3gen.mel2wav.* -> mel2wav.* (snake .alpha -> .weight, convs wrap in .conv).
        for (k, v) in allWeights where k.hasPrefix("s3gen.mel2wav.") {
            var key = "mel2wav." + String(k.dropFirst("s3gen.mel2wav.".count))
            if key.hasSuffix(".alpha") {
                key += ".weight"
                weights[key] = v.reshaped([v.dim(0), 1])
            } else if key.contains("f0_predictor.condnet.") && (key.hasSuffix(".weight") || key.hasSuffix(".bias")) {
                let base = String(key.dropLast((key.hasSuffix(".weight") ? ".weight" : ".bias").count))
                weights[base + ".conv." + (key.hasSuffix(".weight") ? "weight" : "bias")] = v
            } else if key.hasPrefix("mel2wav.conv_pre.") || key.hasPrefix("mel2wav.conv_post.")
                        || key.hasPrefix("mel2wav.ups.") || key.hasPrefix("mel2wav.source_downs.")
                        || key.hasPrefix("mel2wav.resblocks.") || key.hasPrefix("mel2wav.source_resblocks.") {
                // Conv1dPT/ConvTranspose1dPT/HifiResBlock wrap convs in `.conv`.
                let dot = key.lastIndex(of: ".")!
                let base = String(key[..<dot])
                let leaf = String(key[key.index(after: dot)...])
                weights[base + ".conv." + leaf] = v
            } else {
                weights[key] = v
            }
        }

        let model = ChatterboxML(config: config)

        // Per-tensor group sizes keyed by module-tree path.
        let groupSizeForPath = ChatterboxLoader.groupSizeForPath(weights).reduce(into: [String: Int]()) {
            $0[$1.key] = $1.value
        }
        let bitsForPath = ChatterboxLoader.bitsForPath(weights).reduce(into: [String: Int]()) {
            $0[$1.key] = $1.value
        }

        // Dequantize embedding weights (plain Embedding modules hold float weights).
        let embedPaths = weights.keys.filter {
            groupSizeForPath[$0] != nil && weights[$0 + ".weight"] != nil
                && weights[$0 + ".weight"]!.dtype == .uint32
                && weights[$0 + ".scales"] != nil
                && !weights.keys.contains($0 + ".weight") // placeholder, replaced below
        }
        _ = embedPaths
        let dequantEmbeds = ["t3.text_emb", "t3.speech_emb", "t3.text_pos_emb.emb",
                             "t3.speech_pos_emb.emb", "flow.input_embedding"]
        for path in dequantEmbeds {
            guard let packed = weights[path + ".weight"],
                  let sc = weights[path + ".scales"] else { continue }
            let bi = weights[path + ".biases"]
            let g = groupSizeForPath[path] ?? 64
            weights[path + ".weight"] = MLX.dequantized(
                packed, scales: sc, biases: bi, groupSize: g, bits: 4, mode: .affine)
            weights.removeValue(forKey: path + ".scales")
            weights.removeValue(forKey: path + ".biases")
        }

        try applyQuantized(model: model, tensors: weights,
                           groupSizeMap: groupSizeForPath, bitsMap: bitsForPath)
        do {
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [])
        } catch {
            print("[ChatterboxML] ⚠️ update failed: \(error)")
            throw error
        }

        // stft_window is a plain let in HiFTGenerator — pour it manually.
        if let win = weights["mel2wav.stft_window"] {
            model.mel2wav.setStftWindow(win)
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
                t3CondPromptSpeechTokens: condsW["t3.cond_prompt_speech_tokens"] ?? MLXArray(0),
                genPromptToken: condsW["gen.prompt_token"] ?? MLXArray(0),
                genPromptTokenLen: condsW["gen.prompt_token_len"] ?? MLXArray(0),
                genPromptFeat: condsW["gen.prompt_feat"] ?? MLXArray(0),
                genEmbedding: condsW["gen.embedding"] ?? MLXArray(0))
        }

        print("[ChatterboxML] ✅ loaded multilingual (t3 \(config.hiddenLayers)L + flow + vocoder)")
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

    /// Full text → 24kHz waveform (float samples).
    func generate(text: String, language: String = "hi", temperature: Float = 0.8) -> [Float] {
        let toks = speechTokens(text: text, language: language, temperature: temperature)
        let ref = S3RefML(promptToken: conds.genPromptToken,
                          promptTokenLen: conds.genPromptTokenLen,
                          promptFeat: conds.genPromptFeat,
                          embedding: conds.genEmbedding)
        let mel = flow.inference(token: toks, ref: ref, finalize: false)   // (1, 80, T)
        let wav = mel2wav.generate(mel)                        // [Float] 24kHz
        return wav
    }

    // Baked conds (set by loader).
    var conds: T3MLConds = .empty
}

/// Baked default-voice conditioning from conds.safetensors.
struct T3MLConds {
    var t3SpeakerEmb: MLXArray
    var t3EmotionAdv: MLXArray
    var t3CondPromptSpeechTokens: MLXArray
    var genPromptToken: MLXArray
    var genPromptTokenLen: MLXArray
    var genPromptFeat: MLXArray
    var genEmbedding: MLXArray
    init(t3SpeakerEmb: MLXArray, t3EmotionAdv: MLXArray,
         t3CondPromptSpeechTokens: MLXArray, genPromptToken: MLXArray,
         genPromptTokenLen: MLXArray, genPromptFeat: MLXArray, genEmbedding: MLXArray) {
        self.t3SpeakerEmb = t3SpeakerEmb
        self.t3EmotionAdv = t3EmotionAdv
        self.t3CondPromptSpeechTokens = t3CondPromptSpeechTokens
        self.genPromptToken = genPromptToken
        self.genPromptTokenLen = genPromptTokenLen
        self.genPromptFeat = genPromptFeat
        self.genEmbedding = genEmbedding
    }
    static let empty = T3MLConds(t3SpeakerEmb: MLXArray(0), t3EmotionAdv: MLXArray(0),
                                 t3CondPromptSpeechTokens: MLXArray(0),
                                 genPromptToken: MLXArray(0), genPromptTokenLen: MLXArray(0),
                                 genPromptFeat: MLXArray(0), genEmbedding: MLXArray(0))
}
