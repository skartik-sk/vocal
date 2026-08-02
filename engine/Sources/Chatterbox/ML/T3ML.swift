//
//  T3ML.swift — multilingual T3 (text → speech tokens) for the non-turbo
//  Chatterbox (chatterbox-4bit). Ported from mlx-audio chatterbox/t3/t3.py
//  + tokenizer.py (MTLTokenizer).
//
//  Differences vs the turbo T3:
//    - Llama backbone (RoPE, RMSNorm, SwiGLU) instead of GPT-2.
//    - Learned text/speech position embeddings.
//    - A multilingual tokenizer that prepends a language tag (e.g. "[hi]")
//      and replaces spaces with "[SPACE]".
//    - Conditioning: speaker emb + perceiver-resampled prompt + emotion.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// MARK: - MTL (multilingual) text tokenizer

/// Multilingual tokenizer mirroring mlx-audio's MTLTokenizer: NFKD-normalize,
/// prepend the language tag, and encode with spaces → "[SPACE]".
final class MTLTokenizer {
    private let bpe: BPE
    private let spaceToken: String = "[SPACE]"

    init(_ bpe: BPE) {
        self.bpe = bpe
    }

    public func tokenize(text: String, languageID: String?) -> [Int] {
        var txt = text
        // NFKD normalize + lowercase (matches Python preprocess_text)
        txt = txt.applyingTransform(StringTransform("NFKD; Lowercase"), reverse: false) ?? txt
        if let lang = languageID {
            txt = "[\(lang.lowercased())]" + txt
        }
        txt = txt.replacingOccurrences(of: " ", with: spaceToken)
        return bpe.encode(text: txt)
    }
}

// MARK: - Conditioning

/// The baked default-voice conditioning for the multilingual model.
public struct T3MLCond {
    public let speakerEmb: MLXArray            // [1, 256]
    public let emotionAdv: MLXArray            // [1, 1, 1]
    public let condPromptSpeechTokens: MLXArray // [1, 150]
}

/// Conditioning encoder: speaker → 1024, emotion → 1024, perceiver prompt.
final class T3MLCondEnc: Module {
    let speakerEmbedSize: Int
    @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear
    @ModuleInfo(key: "emotion_adv_fc") var emotionFc: Linear
    @ModuleInfo(key: "perceiver") var perceiver: Perceiver

    init(config: LlamaT3Config) {
        self.speakerEmbedSize = config.speakerEmbedSize
        _spkrEnc.wrappedValue = Linear(config.speakerEmbedSize, config.hiddenSize)
        _emotionFc.wrappedValue = Linear(1, config.hiddenSize, bias: false)
        _perceiver.wrappedValue = Perceiver()
        super.init()
    }

    func callAsFunction(_ cond: T3MLCond, speechEmb: (MLXArray) -> MLXArray,
                        speechPosEmb: (MLXArray) -> MLXArray) -> MLXArray {
        let B = cond.speakerEmb.dim(0)
        // speaker → [B,1,1024]
        let condSpkr = spkrEnc(cond.speakerEmb.reshaped([B, speakerEmbedSize]))
            .expandedDimensions(axis: 1)
        // prompt speech tokens → emb + pos, then perceiver-resample
        let promptEmb = speechEmb(cond.condPromptSpeechTokens)
            + speechPosEmb(cond.condPromptSpeechTokens)
        let condPrompt = perceiver(promptEmb)                     // [B,32,1024]
        // emotion → [B,1,1024]
        let emotionVal = cond.emotionAdv.reshaped([B, 1, 1])
        let condEmotion = emotionFc(emotionVal)                   // [B,1,1024]
        return concatenated([condSpkr, condPrompt, condEmotion], axis: 1)  // [B,34,1024]
    }
}

// MARK: - Perceiver resampler

/// Perceiver resampler: cross-attention then self-attention to fixed length.
final class Perceiver: Module {
    @ModuleInfo(key: "pre_attention_query") var preAttentionQuery: MLXArray
    @ModuleInfo(key: "attn") var attn: PerceiverAttentionBlock

    init(preAttentionQueryToken: Int = 32, preAttentionQuerySize: Int = 1024,
         embeddingDim: Int = 1024, numAttnHeads: Int = 4) {
        let queryVariance = Float(sqrt(3.0) * sqrt(2.0 / Double(preAttentionQueryToken + preAttentionQueryToken)))
        _preAttentionQuery.wrappedValue = MLXRandom.uniform(
            low: -queryVariance, high: queryVariance,
            [1, preAttentionQueryToken, preAttentionQuerySize])
        _attn.wrappedValue = PerceiverAttentionBlock(embeddingDim, numAttnHeads)
        super.init()
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let B = h.dim(0)
        let query = broadcast(preAttentionQuery, to: [B, preAttentionQuery.dim(1), preAttentionQuery.dim(2)])
        let preAtt = attn(query, h)        // cross-attention
        return attn(preAtt, preAtt)        // self-attention
    }
}

/// Cross-attention block with separate Q/K/V (perceiver).
final class PerceiverAttentionBlock: Module {
    let numHeads: Int
    let headDim: Int
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ channels: Int, _ numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = channels / numHeads
        _norm.wrappedValue = LayerNorm(dimensions: channels)
        _toQ.wrappedValue = Linear(channels, channels)
        _toK.wrappedValue = Linear(channels, channels)
        _toV.wrappedValue = Linear(channels, channels)
        _projOut.wrappedValue = Linear(channels, channels)
        super.init()
    }

    func callAsFunction(_ x1: MLXArray, _ x2: MLXArray) -> MLXArray {
        let q = toQ(norm(x1)).reshaped([x1.dim(0), x1.dim(1), numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = toK(norm(x2)).reshaped([x2.dim(0), x2.dim(1), numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = toV(norm(x2)).reshaped([x2.dim(0), x2.dim(1), numHeads, headDim]).transposed(0, 2, 1, 3)
        let scale = pow(Float(headDim), -0.5)
        let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let attn = softmax(scores, axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped([x1.dim(0), x1.dim(1), -1])
        return x1 + projOut(out)
    }
}

// MARK: - Multilingual T3

/// Multilingual T3: text tokens → speech tokens (Llama backbone).
final class T3ML: Module {
    let config: LlamaT3Config
    @ModuleInfo(key: "tfmr") var tfmr: LlamaT3Backbone
    @ModuleInfo(key: "cond_enc") var condEnc: T3MLCondEnc
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") var speechEmb: Embedding
    @ModuleInfo(key: "text_pos_emb") var textPosEmb: LearnedPosEmb
    @ModuleInfo(key: "speech_pos_emb") var speechPosEmb: LearnedPosEmb
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "speech_head") var speechHead: Linear

    init(config: LlamaT3Config) {
        self.config = config
        _tfmr.wrappedValue = LlamaT3Backbone(config)
        _condEnc.wrappedValue = T3MLCondEnc(config: config)
        _textEmb.wrappedValue = Embedding(embeddingCount: config.textTokensDictSize, dimensions: config.hiddenSize)
        _speechEmb.wrappedValue = Embedding(embeddingCount: config.speechTokensDictSize, dimensions: config.hiddenSize)
        _textPosEmb.wrappedValue = LearnedPosEmb(maxLen: 2050, dim: config.hiddenSize)
        _speechPosEmb.wrappedValue = LearnedPosEmb(maxLen: 4100, dim: config.hiddenSize)
        _textHead.wrappedValue = Linear(config.hiddenSize, config.textTokensDictSize, bias: false)
        _speechHead.wrappedValue = Linear(config.hiddenSize, config.speechTokensDictSize, bias: false)
        super.init()
    }

    func prepareConditioning(_ cond: T3MLCond) -> MLXArray {
        condEnc(cond, speechEmb: { self.speechEmb($0) },
                speechPosEmb: { self.speechPosEmb($0) })
    }

    /// Build [cond | text | speech] input embeddings.
    func prepareInputEmbeds(cond: T3MLCond, textTokens: MLXArray, speechTokens: MLXArray,
                            cfgWeight: Float) -> (MLXArray, Int) {
        let condEmb = prepareConditioning(cond)                    // [1,34,1024]
        var textEmbOut = textEmb(textTokens)
        var speechEmbOut = speechEmb(speechTokens)
        if cfgWeight > 0 && textEmbOut.dim(0) > 1 {
            let zeros = MLXArray.zeros(textEmbOut[1..<2].shape)
            textEmbOut = concatenated([textEmbOut[0..<1], zeros], axis: 0)
        }
        textEmbOut = textEmbOut + textPosEmb(textTokens)
        speechEmbOut = speechEmbOut + speechPosEmb(speechTokens)
        let lenCond = condEmb.dim(1)
        // CFG: duplicate the whole [cond | text | speech] batch to B=2 with the
        // second item un-conditioned (zeroed text). Matches Python's inference.
        var condEmbFinal = condEmb
        if cfgWeight > 0 && condEmb.dim(0) == 1 && textEmbOut.dim(0) == 1 {
            condEmbFinal = concatenated([condEmb, condEmb], axis: 0)
            textEmbOut = concatenated([textEmbOut, MLXArray.zeros(textEmbOut.shape)], axis: 0)
            speechEmbOut = concatenated([speechEmbOut, speechEmbOut], axis: 0)
        }
        let embeds = concatenated([condEmbFinal, textEmbOut, speechEmbOut], axis: 1)
        return (embeds, lenCond)
    }

    /// Autoregressive text → speech-token generation.
    func inference(cond: T3MLCond, textTokens: MLXArray,
                   maxNewTokens: Int = 1024, temperature: Float = 0.8,
                   topP: Float = 0.95, minP: Float = 0.05,
                   repetitionPenalty: Float = 1.2, cfgWeight: Float = 0.5,
                   greedy: Bool = false) -> MLXArray {
        let B = textTokens.dim(0)
        let bosToken = MLXArray([Int32(config.startSpeechToken)]).reshaped([1, 1])
        let cache: [KVCache] = (0 ..< config.hiddenLayers).map { _ in KVCacheSimple() }

        // Initial prefill: [cond | text | bos]
        let (embeds, _) = prepareInputEmbeds(
            cond: cond, textTokens: textTokens, speechTokens: bosToken, cfgWeight: cfgWeight)
        var hidden = tfmr(embeds, cache: cache)
        let lastIdx = hidden.dim(1) - 1
        var logits0 = speechHead(hidden[0..., lastIdx, 0...])       // [B, vocab]
        if cfgWeight > 0 && logits0.dim(0) > 1 {
            let c = logits0[0..<1]
            let u = logits0[1..<2]
            logits0 = c + cfgWeight * (c - u)
        }
        // speechHead(hidden[..., last, ...]) with an integer index drops the time dim:
        // logits0 is already [B, vocab].
        let logits0Flat = logits0

        var generated = [Int]()
        var nextToken = sample(logits0Flat, temperature: temperature, topP: topP,
                               minP: minP, generated: generated,
                               repetitionPenalty: repetitionPenalty, greedy: greedy)
        var nextTokenId = Int(nextToken[0].item(Int32.self))
        // Match Python: generated_ids starts with the BOS speech token, then appends.
        generated.append(config.startSpeechToken)
        generated.append(nextTokenId)

        // Generation loop
        for step in 0 ..< maxNewTokens {
            if nextTokenId == config.stopSpeechToken { break }
            let tokenArr = MLXArray([Int32(nextTokenId)]).reshaped([1, 1])
            var nextEmb = speechEmb(tokenArr) + speechPosEmb.fixed(step + 1)
            if cfgWeight > 0 {
                nextEmb = concatenated([nextEmb, nextEmb], axis: 0)
            }
            hidden = tfmr(nextEmb, cache: cache)
            var logits = speechHead(hidden[0..., hidden.dim(1) - 1, 0...])  // [B, vocab]
            if cfgWeight > 0 && logits.dim(0) > 1 {
                let c = logits[0..<1]
                let u = logits[1..<2]
                logits = c + cfgWeight * (c - u)
            }
            logits = logits[0..<1, 0...]
            nextToken = sample(logits, temperature: temperature, topP: topP, minP: minP,
                               generated: generated, repetitionPenalty: repetitionPenalty,
                               greedy: greedy)
            nextTokenId = Int(nextToken[0].item(Int32.self))
            generated.append(nextTokenId)
            if generated.count >= 2 && generated.last == config.stopSpeechToken { break }
            if generated.count >= maxNewTokens { break }
        }

        // drop a trailing EOS if present
        if generated.last == config.stopSpeechToken { generated.removeLast() }
        return MLXArray(generated.map { Int32($0) }).reshaped([1, -1])
    }

    private func sample(_ logits: MLXArray, temperature: Float, topP: Float, minP: Float,
                        generated: [Int], repetitionPenalty: Float, greedy: Bool) -> MLXArray {
        var lg = logits
        if repetitionPenalty != 1.0 && !generated.isEmpty {
            lg = applyRepetitionPenalty(lg, generated: generated, penalty: repetitionPenalty)
        }
        if greedy {
            return argMax(lg, axis: -1).reshaped([logits.dim(0), 1])
        }
        if temperature > 0 && temperature != 1.0 { lg = lg / temperature }
        if topP < 1.0 { lg = topPFilter(lg, topP) }
        if minP > 0.0 { lg = minPFilter(lg, minP) }
        return MLXRandom.categorical(lg).reshaped([logits.dim(0), 1])
    }

    private func applyRepetitionPenalty(_ logits: MLXArray, generated: [Int], penalty: Float) -> MLXArray {
        let V = logits.dim(-1)
        var seen = Set<Int>()
        for g in generated where g >= 0 && g < V { seen.insert(g) }
        if seen.isEmpty { return logits }
        var maskSwift = [Float](repeating: 0, count: V)
        for g in seen { maskSwift[g] = 1 }
        let mask = MLXArray(maskSwift).reshaped([1, V])
        let penalized = MLX.where(logits .< 0, logits * penalty, logits / penalty)
        return MLX.where(mask .> 0, penalized, logits)
    }

    private func topPFilter(_ logits: MLXArray, _ topP: Float) -> MLXArray {
        let order = argSort(-logits, axis: -1)
        let sortedLogits = takeAlong(logits, order, axis: -1)
        let sortedProbs = softmax(sortedLogits, axis: -1)
        let cum = sortedProbs.cumsum(axis: -1)
        var remove = cum .> topP
        let B = logits.dim(0)
        remove = concatenated([MLXArray.zeros([B, 1], dtype: .bool), remove[0..., 0..<(cum.dim(-1) - 1)]], axis: -1)
        let maskedSorted = MLX.where(remove, MLXArray(-Float.infinity), sortedLogits)
        let inverse = argSort(order, axis: -1)
        return takeAlong(maskedSorted, inverse, axis: -1)
    }

    private func minPFilter(_ logits: MLXArray, _ minP: Float) -> MLXArray {
        let probs = softmax(logits, axis: -1)
        let topProb = max(probs, axis: -1, keepDims: true)
        let mask = probs .>= (topProb * minP)
        return MLX.where(mask, logits, MLXArray(-Float.infinity))
    }
}

/// Learned position embeddings (indexed by SEQUENCE POSITION, not token id —
/// matches Python's LearnedPositionEmbeddings which uses arange over the length).
final class LearnedPosEmb: Module {
    @ModuleInfo(key: "emb") var emb: Embedding
    init(maxLen: Int, dim: Int) {
        _emb.wrappedValue = Embedding(embeddingCount: maxLen, dimensions: dim)
        super.init()
    }
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let sl = tokens.dim(1)
        return emb(MLXArray((0 ..< sl).map { Int32($0) }).reshaped([1, sl]))
    }
    func fixed(_ idx: Int) -> MLXArray {
        emb(MLXArray([Int32(idx)]).reshaped([1, 1]))
    }
}
