//
//  T3.swift — Token-To-Token TTS: text → discrete speech tokens (GPT-2 backbone).
//
//  Ported from chatterbox_turbo/models/t3/{t3,cond_enc}.py. Turbo path: single-stream,
//  no classifier-free guidance. Conditioned on the baked default voice (speaker_emb +
//  375-token cond prompt from conds.safetensors).
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Container for T3 conditioning (the baked default-voice tensors).
struct T3Cond {
    let speakerEmb: MLXArray             // [1, 256]
    let condPromptSpeechTokens: MLXArray // [1, 375]
}

/// Non-text conditioning: projects speaker emb, prepends the voice prompt. Turbo ignores
/// CLAP/emotion (empty tensors), so this reduces to [speaker, cond_prompt].
final class T3CondEnc: Module {
    @ModuleInfo(key: "spkr_enc") private(set) var spkrEnc: Linear

    init(speakerEmbedSize: Int, dim: Int) {
        self._spkrEnc.wrappedValue = Linear(speakerEmbedSize, dim)
        super.init()
    }

    func callAsFunction(_ cond: T3Cond, speechEmb: (MLXArray) -> MLXArray) -> MLXArray {
        let s = cond.speakerEmb.reshaped([-1, cond.speakerEmb.dim(-1)])
        let condSpkr = spkrEnc(s).expandedDimensions(axis: 1)      // [B, 1, dim]
        let condPrompt = speechEmb(cond.condPromptSpeechTokens)    // [B, 375, dim]
        ChatterboxDump(condSpkr, "swift_condSpkr")
        ChatterboxDump(condPrompt, "swift_condPrompt")
        return concatenated([condSpkr, condPrompt], axis: 1)       // [B, 376, dim]
    }
}

final class T3: Module {
    let hp: T3ConfigSection
    let dim: Int
    @ModuleInfo(key: "tfmr") private(set) var tfmr: GPT2Model
    @ModuleInfo(key: "cond_enc") private(set) var condEnc: T3CondEnc
    @ModuleInfo(key: "text_emb") private(set) var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") private(set) var speechEmb: Embedding
    @ModuleInfo(key: "text_head") private(set) var textHead: Linear
    @ModuleInfo(key: "speech_head") private(set) var speechHead: Linear

    init(gpt2: GPT2Config, hp: T3ConfigSection) {
        self.hp = hp
        self.dim = gpt2.nEmbd
        self._tfmr.wrappedValue = GPT2Model(gpt2)
        self._condEnc.wrappedValue = T3CondEnc(speakerEmbedSize: hp.speakerEmbedSize, dim: gpt2.nEmbd)
        self._textEmb.wrappedValue = Embedding(embeddingCount: hp.textTokensDictSize, dimensions: gpt2.nEmbd)
        self._speechEmb.wrappedValue = Embedding(embeddingCount: hp.speechTokensDictSize, dimensions: gpt2.nEmbd)
        self._textHead.wrappedValue = Linear(gpt2.nEmbd, hp.textTokensDictSize, bias: false)
        self._speechHead.wrappedValue = Linear(gpt2.nEmbd, hp.speechTokensDictSize, bias: true)
        super.init()
    }

    /// Build the full input embeds: [cond | text | speech_start].
    private func prepareInputEmbeds(cond: T3Cond, textTokens: MLXArray, speechStart: MLXArray) -> MLXArray {
        let condEmb = condEnc(cond, speechEmb: { self.speechEmb($0) })
        ChatterboxDump(condEmb, "swift_condemb")
        let textE = textEmb(textTokens)
        let speechE = speechEmb(speechStart)
        let emb = concatenated([condEmb, textE, speechE], axis: 1)
        ChatterboxDump(emb, "swift_embeds")
        return emb
    }

    /// Autoregressive text → speech-token generation (the turbo path).
    func inference(
        cond: T3Cond,
        textTokens: MLXArray,
        temperature: Float = 0.8,
        topK: Int = 1000,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2,
        maxGenLen: Int = 800,
        greedy: Bool = false
    ) -> MLXArray {
        let B = textTokens.dim(0)
        let speechStart = MLXArray(Int32(hp.startSpeechToken)).reshaped([B, 1])
        let cache: [KVCache] = (0..<tfmr.config.nLayer).map { _ in KVCacheSimple() }

        // Prime: process [cond | text | speech_start]; predict the first speech token from
        // the hidden state at the last sequence position (integer index drops the time dim).
        var hidden = tfmr(
            inputsEmbeds: prepareInputEmbeds(cond: cond, textTokens: textTokens, speechStart: speechStart),
            cache: cache)
        let lastIdx = hidden.dim(1) - 1
        let logits0 = speechHead(hidden[0..., lastIdx, 0...])    // [B, vocab]
        ChatterboxDump(logits0, "swift_t3_logits0")
        var nextToken = sample(
            logits0,
            temperature: temperature, topK: topK, topP: topP,
            generated: nil, repetitionPenalty: repetitionPenalty, greedy: greedy)

        var generated: [MLXArray] = [nextToken]
        var current = nextToken

        for _ in 0..<maxGenLen {
            hidden = tfmr(inputsEmbeds: speechEmb(current), cache: cache)
            let allGen = concatenated(generated, axis: 1)  // [B, t]
            nextToken = sample(
                speechHead(hidden[0..., 0, 0...]),         // [B, vocab] (the single new token)
                temperature: temperature, topK: topK, topP: topP,
                generated: allGen, repetitionPenalty: repetitionPenalty, greedy: greedy)
            generated.append(nextToken)
            current = nextToken
            // eval() is MLX's array materializer (forces the lazy graph), not string eval.
            eval(nextToken)
            if Int(nextToken[0].item(Int32.self)) == hp.stopSpeechToken { break }
        }

        var allTokens = concatenated(generated, axis: 1)   // [B, t]
        eval(allTokens)
        // drop a trailing EOS if present
        if allTokens.dim(1) > 0,
           Int(allTokens[0..., allTokens.dim(1) - 1].item(Int32.self)) == hp.stopSpeechToken {
            allTokens = allTokens[0..., 0..<(allTokens.dim(1) - 1)]
        }
        return allTokens
    }

    // MARK: - Sampling

    private func sample(
        _ logits: MLXArray,
        temperature: Float,
        topK: Int,
        topP: Float,
        generated: MLXArray?,
        repetitionPenalty: Float,
        greedy: Bool
    ) -> MLXArray {
        var lg = logits
        if let generated, repetitionPenalty != 1.0 {
            lg = applyRepetitionPenalty(lg, generated: generated, penalty: repetitionPenalty)
        }
        if greedy {
            return argMax(lg, axis: -1).reshaped([logits.dim(0), 1])
        }
        if temperature > 0 && temperature != 1.0 { lg = lg / temperature }
        if topK > 0 { lg = topKFilter(lg, topK) }
        if topP < 1.0 { lg = topPFilter(lg, topP) }
        return MLXRandom.categorical(lg).reshaped([logits.dim(0), 1])
    }

    /// Penalize tokens already generated. Builds a vocab mask in Swift (mlx-swift has no oneHot)
    /// — for generated indices: logit<0 → ×penalty, logit>0 → ÷penalty.
    private func applyRepetitionPenalty(_ logits: MLXArray, generated: MLXArray, penalty: Float) -> MLXArray {
        let V = logits.dim(-1)
        let gens = generated.asArray(Int32.self)
        var seen = Set<Int32>()
        for g in gens where g >= 0 && g < Int32(V) { seen.insert(g) }
        if seen.isEmpty { return logits }
        var maskSwift = [Float](repeating: 0, count: V)
        for g in seen { maskSwift[Int(g)] = 1 }
        let mask = MLXArray(maskSwift).reshaped([1, V])           // [1, V] broadcasts over batch
        let penalized = MLX.where(logits .< 0, logits * penalty, logits / penalty)
        return MLX.where(mask .> 0, penalized, logits)
    }

    private func topKFilter(_ logits: MLXArray, _ topK: Int) -> MLXArray {
        let V = logits.dim(-1)
        let k = min(topK, V)
        let part = partitioned(logits, kth: V - k, axis: -1)
        let threshold = part[0..., (V - k)..<(V - k + 1)]          // [B, 1]
        return MLX.where(logits .>= threshold, logits, MLXArray(-Float.infinity))
    }

    private func topPFilter(_ logits: MLXArray, _ topP: Float) -> MLXArray {
        let order = argSort(-logits, axis: -1)                     // descending order
        let sortedLogits = takeAlong(logits, order, axis: -1)
        let sortedProbs = softmax(sortedLogits, axis: -1)
        let cum = sortedProbs.cumsum(axis: -1)
        var remove = cum .> topP
        let B = logits.dim(0)
        // shift right: keep the first token above the threshold
        remove = concatenated(
            [MLXArray.zeros([B, 1], dtype: .bool), remove[0..., 0..<(cum.dim(-1) - 1)]], axis: -1)
        let maskedSorted = MLX.where(remove, MLXArray(-Float.infinity), sortedLogits)
        let inverse = argSort(order, axis: -1)                     // inverse permutation
        return takeAlong(maskedSorted, inverse, axis: -1)
    }
}
