//
//  S3Gen.swift — speech tokens → mel (S3Token2Mel) via the mean-flow CFM.
//
//  Ported from chatterbox_turbo/models/s3gen/{s3gen,flow_matching}.py. The default-voice path
//  uses the baked conds (prompt_token, prompt_feat, embedding) — no CAMPPlus/voice-encoder.
//  Masks are all-ones: single-utterance inference has no padding (token_len == actual length),
//  so masking is a no-op and the mean-flow math is unchanged.
//

import Foundation
import MLX
import MLXNN

/// The mean-flow CFM wrapper. weight path: `decoder.estimator.*`.
final class S3CFM: Module {
    @ModuleInfo(key: "estimator") var estimator: ConditionalDecoder

    init(meanflow: Bool) {
        _estimator.wrappedValue = ConditionalDecoder(meanflow: meanflow)
        super.init()
    }

    /// 2-step Euler (mean-flow, linear schedule, no CFG). Returns mel (B, 80, T).
    func callAsFunction(
        mu: MLXArray, mask: MLXArray, nTimesteps: Int,
        spks: MLXArray?, cond: MLXArray?, noisedMels: MLXArray?
    ) -> MLXArray {
        var z = MLXRandom.normal(mu.shape)
        if let nm = noisedMels {
            let promptLen = mu.dim(2) - nm.dim(2)
            z = concatenated([z[0..., 0..., 0..<promptLen], nm], axis: 2)
        }
        // linear t_span = linspace(0,1,n+1); meanflow skips the cosine warp
        let ts = (0...nTimesteps).map { Float($0) / Float(nTimesteps) }
        for i in 0..<nTimesteps {
            let t = MLXArray([ts[i]]).reshaped([1])
            let r = MLXArray([ts[i + 1]]).reshaped([1])
            let dxdt = estimator(x: z, mask: mask, mu: mu, t: t, spks: spks, cond: cond, r: r)
            z = z + (r - t) * dxdt
        }
        return z
    }
}

/// Reference/conditioning data for the default voice (from conds.safetensors).
struct S3Ref {
    let promptToken: MLXArray      // [1, 250]
    let promptTokenLen: MLXArray   // [1]
    let promptFeat: MLXArray       // [1, 500, 80]
    let embedding: MLXArray        // [1, 192]
}

/// S3 token→mel generator. weight paths: `input_embedding`, `spk_embed_affine_layer`,
/// `encoder.*`, `encoder_proj`, `decoder.estimator.*`.
final class S3Gen: Module {
    let tokenMelRatio = 2
    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffine: Linear
    @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    var decoder: S3CFM

    init(meanflow: Bool) {
        _inputEmbedding.wrappedValue = Embedding(embeddingCount: 6561, dimensions: 512)
        _spkEmbedAffine.wrappedValue = Linear(192, 80)
        _encoder.wrappedValue = UpsampleConformerEncoder(
            inputSize: 512, outputSize: 512, attentionHeads: 8, linearUnits: 2048, numBlocks: 6)
        _encoderProj.wrappedValue = Linear(512, 80)
        self.decoder = S3CFM(meanflow: meanflow)
        super.init()
    }

    /// speech_tokens (B, T) → mel (B, 80, T_mel).
    func callAsFunction(_ speechTokens: MLXArray, ref: S3Ref) -> MLXArray {
        let B = speechTokens.dim(0)

        // Speaker embedding: L2-normalize, then project to 80.
        let n = (ref.embedding.square()).sum(axis: -1, keepDims: true).sqrt()
        var emb = ref.embedding / (n + 1e-8)
        emb = spkEmbedAffine(emb)                                   // (B, 80)

        let token = concatenated([ref.promptToken, speechTokens], axis: 1)   // (B, 250+T)
        let promptT = ref.promptToken.dim(1)
        let totalT = token.dim(1)
        let ones = MLXArray.ones([B, totalT, 1])
        let tokenEmb = inputEmbedding(token) * ones                 // (B, 250+T, 512)

        let lens = MLXArray([Int32(totalT)])
        let (h, _) = encoder(tokenEmb, xsLens: lens)                // (B, 2*(250+T), 512)

        let melLen1 = ref.promptFeat.dim(1)                          // 500
        let melLen2 = h.dim(1) - melLen1
        var mu = encoderProj(h)                                      // (B, 2*(250+T), 80)

        // Conditioning: [prompt_feat, zeros] -> (B, 80, T)
        let zeros = MLXArray.zeros([B, melLen2, 80])
        let conds = concatenated([ref.promptFeat, zeros], axis: 1).transposed(0, 2, 1)

        let mask = MLXArray.ones([B, 1, mu.dim(1)])
        let noisedMels = MLXRandom.normal([B, 80, speechTokens.dim(1) * 2])

        var feat = decoder(mu: mu.transposed(0, 2, 1), mask: mask, nTimesteps: 2,
                           spks: emb, cond: conds, noisedMels: noisedMels)
        feat = feat[0..., 0..., melLen1...]                         // drop the prompt portion
        _ = mu
        return feat
    }
}
