//
//  MLS3Gen.swift — multilingual S3Gen (speech tokens → mel) for chatterbox-4bit.
//  Ported from mlx-audio chatterbox/s3gen/flow.py + decoder.py (Matcha-style
//  CausalMaskedDiffWithXvec with 10-step Euler CFM).
//
//  Reuses the existing UpsampleConformerEncoder (S3/Encoder.swift) for the
//  conformer encoder; this file adds the Matcha decoder (CausalResnetBlock1D +
//  BasicTransformerBlock) and the CausalMaskedDiffWithXvec flow wrapper.
//

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Causal resnet block (Matcha)

final class MLCausalConv1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let causalPadding: Int
    init(_ inCh: Int, _ outCh: Int, _ kernel: Int) {
        self.causalPadding = kernel - 1
        _conv.wrappedValue = Conv1d(
            inputChannels: inCh, outputChannels: outCh, kernelSize: kernel,
            stride: 1, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T) -> pad -> conv -> (B, C, T)
        let xT = x.transposed(0, 2, 1)  // (B, T, C)
        var paddedX = xT
        if causalPadding > 0 {
            paddedX = padded(xT, widths: [[0, 0], [causalPadding, 0], [0, 0]])
        }
        return conv(paddedX).transposed(0, 2, 1)
    }
}

final class MLCausalBlock1D: Module {
    @ModuleInfo(key: "conv") var conv: MLCausalConv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(_ dim: Int, _ dimOut: Int) {
        _conv.wrappedValue = MLCausalConv1d(dim, dimOut, 3)
        _norm.wrappedValue = LayerNorm(dimensions: dimOut)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = conv(x * mask)
        h = norm(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        return mish(h) * mask
    }
}

/// A conv whose weights live directly at `weight`/`bias` (no nested `.conv`),
/// matching the checkpoint's `res_conv.weight` layout.
final class MLDirectConv1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray
    init(_ inCh: Int, _ outCh: Int, _ kernel: Int) {
        let scale = sqrt(1 / Float(inCh * kernel))
        _weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outCh, kernel, inCh])
        _bias.wrappedValue = MLXArray.zeros([outCh])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T) -> (B, T, C) -> conv -> (B, T, C) -> (B, C, T)
        let xT = x.transposed(0, 2, 1)
        let y = conv1d(xT, weight, padding: 0)
        return (y + bias).transposed(0, 2, 1)
    }
}

/// Matcha CausalResnetBlock1D: block1 -> time-mlp -> block2 -> +res_conv.
final class MLResnetBlock1D: Module {
    @ModuleInfo(key: "block1") var block1: MLCausalBlock1D
    @ModuleInfo(key: "block2") var block2: MLCausalBlock1D
    @ModuleInfo(key: "mlp_linear") var mlpLinear: Linear
    @ModuleInfo(key: "res_conv") var resConv: MLDirectConv1d
    init(_ dim: Int, _ dimOut: Int, _ timeDim: Int) {
        _block1.wrappedValue = MLCausalBlock1D(dim, dimOut)
        _block2.wrappedValue = MLCausalBlock1D(dimOut, dimOut)
        _mlpLinear.wrappedValue = Linear(timeDim, dimOut)
        _resConv.wrappedValue = MLDirectConv1d(dim, dimOut, 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = block1(x, mask: mask)
        h = h + mlpLinear(mish(timeEmb)).expandedDimensions(axis: -1)
        h = block2(h, mask: mask)
        return h + resConv(x * mask)
    }
}

// MARK: - Matcha transformer block

/// Diffusers-style attention: q/k/v project to heads*dim_head (512), out to dim.
final class MLDiffusersAttention: Module {
    let heads: Int, dimHead: Int, scale: Float
    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    init(queryDim: Int, heads: Int, dimHead: Int) {
        self.heads = heads; self.dimHead = dimHead; self.scale = pow(Float(dimHead), -0.5)
        let inner = heads * dimHead
        _queryProj.wrappedValue = Linear(queryDim, inner, bias: false)
        _keyProj.wrappedValue = Linear(queryDim, inner, bias: false)
        _valueProj.wrappedValue = Linear(queryDim, inner, bias: false)
        _outProj.wrappedValue = Linear(inner, queryDim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let hs = [B, T, heads, dimHead]
        let q = queryProj(x).reshaped(hs).transposed(0, 2, 1, 3)
        let k = keyProj(x).reshaped(hs).transposed(0, 2, 1, 3)
        let v = valueProj(x).reshaped(hs).transposed(0, 2, 1, 3)
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale  // (B, H, T, T)
        if let mask {
            // mask: (B, T, T) additive bias
            scores = scores + mask.expandedDimensions(axis: 1)
        } else {
            // Full-context (all positions attend) — Python builds an all-zero
            // additive bias from the length mask; use fast SDPA to match.
            return outProj(
                MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: scale, mask: .none)
                .transposed(0, 2, 1, 3).reshaped([B, T, -1]))
        }
        let attn = softmax(scores, axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped([B, T, -1])
        return outProj(out)
    }
}

/// Feed-forward with GELU (ff.layers.0/1).
final class MLFeedForward: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]
    init(_ dim: Int, _ mult: Int) {
        _layers.wrappedValue = [Linear(dim, mult), Linear(mult, dim)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = layers[0](x)
        return layers[1](gelu(h))
    }
}

/// BasicTransformerBlock: norm1 -> attn -> + -> norm3 -> ff -> +.
final class MLBasicTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: MLDiffusersAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: MLFeedForward
    init(_ dim: Int, heads: Int, dimHead: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = MLDiffusersAttention(queryDim: dim, heads: heads, dimHead: dimHead)
        _norm3.wrappedValue = LayerNorm(dimensions: dim)
        _ff.wrappedValue = MLFeedForward(dim, dim * 4)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attn(norm1(x), mask: mask)
        h = h + ff(norm3(h))
        return h
    }
}

// MARK: - Down / Mid / Up blocks

final class MLDownBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: MLResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [MLBasicTransformerBlock]
    @ModuleInfo(key: "downsample") var downsample: Module
    init(inputCh: Int, outputCh: Int, timeDim: Int, nBlocks: Int, numHeads: Int, headDim: Int, isLast: Bool) {
        _resnet.wrappedValue = MLResnetBlock1D(inputCh, outputCh, timeDim)
        _transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            MLBasicTransformerBlock(outputCh, heads: numHeads, dimHead: headDim)
        }
        _downsample.wrappedValue = isLast ? MLCausalConv1d(outputCh, outputCh, 3) : Downsample1D(outputCh)
        super.init()
    }
}

final class MLMidBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: MLResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [MLBasicTransformerBlock]
    init(_ ch: Int, _ timeDim: Int, nBlocks: Int, numHeads: Int, headDim: Int) {
        _resnet.wrappedValue = MLResnetBlock1D(ch, ch, timeDim)
        _transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            MLBasicTransformerBlock(ch, heads: numHeads, dimHead: headDim)
        }
        super.init()
    }
}

final class MLUpBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: MLResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [MLBasicTransformerBlock]
    @ModuleInfo(key: "upsample") var upsample: Module
    init(inputCh: Int, outputCh: Int, timeDim: Int, nBlocks: Int, numHeads: Int, headDim: Int, isLast: Bool) {
        _resnet.wrappedValue = MLResnetBlock1D(inputCh, outputCh, timeDim)
        _transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            MLBasicTransformerBlock(outputCh, heads: numHeads, dimHead: headDim)
        }
        _upsample.wrappedValue = isLast ? MLCausalConv1d(outputCh, outputCh, 3) : Upsample1D(outputCh)
        super.init()
    }
}

// MARK: - ConditionalDecoder (Matcha)

/// Matcha-style ConditionalDecoder with time embedding, down/mid/up blocks.
final class MLConditionalDecoder: Module {
    let inChannels: Int
    @ModuleInfo(key: "time_mlp") var timeMlp: TimestepEmbedding
    @ModuleInfo(key: "down_blocks") var downBlocks: [MLDownBlock]
    @ModuleInfo(key: "mid_blocks") var midBlocks: [MLMidBlock]
    @ModuleInfo(key: "up_blocks") var upBlocks: [MLUpBlock]
    @ModuleInfo(key: "final_block") var finalBlock: MLCausalBlock1D
    @ModuleInfo(key: "final_proj") var finalProj: Conv1dPT

    init(inChannels: Int = 320, outChannels: Int = 80, channels: [Int] = [256],
         attentionHeadDim: Int = 64, nBlocks: Int = 4, numMidBlocks: Int = 12,
         numHeads: Int = 8) {
        self.inChannels = inChannels
        let timeDim = channels[0] * 4
        _timeMlp.wrappedValue = TimestepEmbedding(inChannels, timeDim)

        var db = [MLDownBlock]()
        var outputCh = inChannels
        for (i, ch) in channels.enumerated() {
            let inputCh = outputCh; outputCh = ch
            db.append(MLDownBlock(inputCh: inputCh, outputCh: outputCh, timeDim: timeDim,
                                  nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim,
                                  isLast: i == channels.count - 1))
        }
        _downBlocks.wrappedValue = db

        _midBlocks.wrappedValue = (0..<numMidBlocks).map { _ in
            MLMidBlock(channels.last!, timeDim, nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim)
        }

        let channelsUp = Array(channels.reversed()) + [channels[0]]
        var ub = [MLUpBlock]()
        for i in 0..<(channelsUp.count - 1) {
            let inputCh = channelsUp[i] * 2
            ub.append(MLUpBlock(inputCh: inputCh, outputCh: channelsUp[i + 1], timeDim: timeDim,
                                nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim,
                                isLast: i == channelsUp.count - 2))
        }
        _upBlocks.wrappedValue = ub

        _finalBlock.wrappedValue = MLCausalBlock1D(channelsUp.last!, channelsUp.last!)
        _finalProj.wrappedValue = Conv1dPT(channelsUp.last!, outChannels, kernel: 1)
        super.init()
    }

    func callAsFunction(x: MLXArray, mask: MLXArray, mu: MLXArray, t: MLXArray,
                        spks: MLXArray?, cond: MLXArray?) -> MLXArray {
        let tEmb = timeMlp(sinusoidalPosEmb(t, inChannels))
        var inputs = [x, mu]
        if let spks {
            inputs.append(broadcast(spks.expandedDimensions(axis: -1),
                                    to: [spks.dim(0), spks.dim(1), x.dim(2)]))
        }
        if let cond { inputs.append(cond) }
        var h = concatenated(inputs, axis: 1)   // (B, in+mu+spks+cond, T)

        var hiddens = [MLXArray]()
        var masks = [mask]
        for db in downBlocks {
            let maskDown = masks[masks.count - 1]
            h = db.resnet(h, mask: maskDown, timeEmb: tEmb)
            h = h.transposed(0, 2, 1)
            let maskT = maskDown[0..., 0..<1, 0...].squeezed(axis: 1)
            for tb in db.transformerBlocks { h = tb(h, mask: nil) }
            h = h.transposed(0, 2, 1)
            hiddens.append(h)
            if let d = db.downsample as? Downsample1D { h = d(h * maskDown) }
            else if let c = db.downsample as? MLCausalConv1d { h = c(h * maskDown) }
            masks.append(strided(maskDown, axis: 2, by: 2))
        }
        masks.removeLast()
        let maskMid = masks[masks.count - 1]

        for mb in midBlocks {
            h = mb.resnet(h, mask: maskMid, timeEmb: tEmb)
            h = h.transposed(0, 2, 1)
            for tb in mb.transformerBlocks { h = tb(h, mask: nil) }
            h = h.transposed(0, 2, 1)
        }

        var maskUp = maskMid
        for ub in upBlocks {
            maskUp = masks.removeLast()
            let skip = hiddens.removeLast()
            h = h[0..., 0..., 0..<skip.dim(2)]
            h = concatenated([h, skip], axis: 1)
            h = ub.resnet(h, mask: maskUp, timeEmb: tEmb)
            h = h.transposed(0, 2, 1)
            for tb in ub.transformerBlocks { h = tb(h, mask: nil) }
            h = h.transposed(0, 2, 1)
            if let u = ub.upsample as? Upsample1D { h = u(h * maskUp) }
            else if let c = ub.upsample as? MLCausalConv1d { h = c(h * maskUp) }
        }

        h = finalBlock(h, mask: maskUp)
        h = finalProj(h * maskUp)
        return h * mask
    }
}

// MARK: - CFM (CausalConditionalCFM, 10-step Euler with CFG)

/// Deterministic-noise Euler CFM matching CausalConditionalCFM.
final class MLCFM: Module {
    let inChannels: Int
    let spkEmbDim: Int
    let inferenceCfgRate: Float
    @ModuleInfo(key: "estimator") var estimator: MLConditionalDecoder
    @ModuleInfo(key: "rand_noise") var randNoise: MLXArray

    init(inChannels: Int = 320, spkEmbDim: Int = 80, inferenceCfgRate: Float = 0.7) {
        self.inChannels = inChannels
        self.spkEmbDim = spkEmbDim
        self.inferenceCfgRate = inferenceCfgRate
        _estimator.wrappedValue = MLConditionalDecoder(inChannels: 320, outChannels: 80,
                                                       channels: [256], nBlocks: 4,
                                                       numMidBlocks: 12, numHeads: 8)
        // Deterministic noise matching Python: mx.random.seed(0);
        // rand_noise = mx.random.normal((1, 80, 50*300)).
        let key = MLXRandom.key(UInt64(0))
        _randNoise.wrappedValue = MLXRandom.normal([1, 80, 15000], key: key)
        super.init()
    }

    func callAsFunction(mu: MLXArray, mask: MLXArray, nTimesteps: Int,
                        spks: MLXArray?, cond: MLXArray?) -> MLXArray {
        let T = mu.dim(2)
        var z = randNoise[0..., 0..., 0..<T]  // (1, 80, T)

        // cosine t_span: t = 1 - cos(0.5*pi*t)
        let steps = nTimesteps + 1
        var tSpan = (0..<steps).map { Float($0) / Float(nTimesteps) }
        tSpan = tSpan.map { 1 - cos($0 * 0.5 * Float.pi) }

        var t = MLXArray([tSpan[0]]).reshaped([1])
        var dt = tSpan[1] - tSpan[0]

        let B = mu.dim(0)
        let TLen = mu.dim(2)
        for step in 1..<steps {
            // CFG: duplicate batch, uncond mu=0 / spks=0 / cond=0
            var xIn = concatenated([z, z], axis: 0)
            var maskIn = concatenated([mask, mask], axis: 0)
            var muIn = concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)
            var spksIn = MLXArray.zeros([2 * B, spkEmbDim])
            var condIn = MLXArray.zeros([2 * B, 80, TLen])   // mel cond is 80ch
            if let spks { spksIn = concatenated([spks, MLXArray.zeros(spks.shape)], axis: 0) }
            if let cond { condIn = concatenated([cond, MLXArray.zeros(cond.shape)], axis: 0) }
            let dphi = estimator(x: xIn, mask: maskIn, mu: muIn, t: tIn, spks: spksIn, cond: condIn)
            let dCond = dphi[0..<B]
            let dUncond = dphi[B..<(2 * B)]
            let d = (1 + inferenceCfgRate) * dCond - inferenceCfgRate * dUncond
            z = z + dt * d
            t = t + dt
            if step < steps - 1 { dt = tSpan[step + 1] - Float(t.item(Float.self)) }
            xIn = z; maskIn = mask  // reuse (unused after)
        }
        return z
    }
}

// MARK: - Flow wrapper (CausalMaskedDiffWithXvec)

/// The non-turbo flow: embed tokens, conformer-encode, Euler-CFM to mel.
final class MLFlow: Module {
    let vocabSize: Int
    let nTimesteps: Int
    let preLookaheadLen: Int
    let tokenMelRatio: Int
    let outputSize: Int
    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffine: Linear
    @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "decoder") var decoder: MLCFM

    init(vocabSize: Int = 6561, inputSize: Int = 512, outputSize: Int = 80,
         spkEmbedDim: Int = 192, nTimesteps: Int = 10, preLookaheadLen: Int = 3,
         tokenMelRatio: Int = 2) {
        self.vocabSize = vocabSize
        self.nTimesteps = nTimesteps
        self.preLookaheadLen = preLookaheadLen
        self.tokenMelRatio = tokenMelRatio
        self.outputSize = outputSize
        _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: inputSize)
        _spkEmbedAffine.wrappedValue = Linear(spkEmbedDim, outputSize)
        _encoder.wrappedValue = UpsampleConformerEncoder(
            inputSize: 512, outputSize: 512, attentionHeads: 8, linearUnits: 2048, numBlocks: 6)
        _encoderProj.wrappedValue = Linear(512, outputSize)
        _decoder.wrappedValue = MLCFM(inChannels: 320, spkEmbDim: outputSize)
        super.init()
    }

    /// token: (1, T) new tokens; ref: prompt tokens + mel + xvector.
    func inference(token: MLXArray, ref: S3RefML, finalize: Bool = true,
                   nTimesteps: Int? = nil) -> MLXArray {
        let steps = nTimesteps ?? self.nTimesteps
        // speaker embed: normalize + affine (1, 192) -> (1, 80)
        let n = norm(ref.embedding, axes: [1], keepDims: true) + 1e-8
        let spk = spkEmbedAffine(ref.embedding / n)

        // concat prompt + new tokens
        let fullToken = concatenated([ref.promptToken, token], axis: 1)
        let tokenLen = ref.promptTokenLen + MLXArray([token.dim(1)])

        let maxLen = Int(tokenLen.max().item(Int32.self))
        let seqRange = MLXArray((0..<maxLen).map { Int32($0) }).reshaped([1, maxLen])
        var mask = (seqRange .< tokenLen.expandedDimensions(axis: -1)).asType(.float32)
        mask = mask.expandedDimensions(axis: -1)   // (1, maxLen, 1)

        var tok = clip(fullToken, min: 0, max: vocabSize - 1)
        tok = inputEmbedding(tok) * mask

        let (hRaw, _) = encoder(tok, xsLens: tokenLen)
        var h = hRaw
        if !finalize {
            h = h[0..., 0..<(h.dim(1) - preLookaheadLen * tokenMelRatio), 0...]
        }
        let melLen1 = ref.promptFeat.dim(1)
        let melLen2 = h.dim(1) - ref.promptFeat.dim(1)
        h = encoderProj(h)

        // conds: prompt mel front-padded
        var conds = MLXArray.zeros([1, melLen1 + melLen2, outputSize], dtype: h.dtype)
        conds[0..., 0..<melLen1, 0...] = ref.promptFeat
        conds = conds.transposed(0, 2, 1)   // (1, 80, T)

        let totalLen = melLen1 + melLen2
        let cmask = MLXArray.ones([1, 1, totalLen], dtype: h.dtype)

        // CFM solve (10 steps, deterministic noise)
        let feat = decoder(mu: h.transposed(0, 2, 1), mask: cmask, nTimesteps: steps,
                           spks: spk, cond: conds)
        return feat[0..., 0..., melLen1..<totalLen]
    }
}

/// Reference conditioning for the multilingual flow (from conds.safetensors).
struct S3RefML {
    let promptToken: MLXArray      // (1, 157)
    let promptTokenLen: MLXArray   // (1,)
    let promptFeat: MLXArray       // (1, 314, 80)
    let embedding: MLXArray        // (1, 192)
}
