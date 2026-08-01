//
//  Encoder.swift — UpsampleConformerEncoder (speech tokens → encoded features `mu`).
//
//  Ported from chatterbox_turbo/models/s3gen/encoder.py. Operates in MLX-native (B, T, C).
//  Uses ESPnet-style relative positional encoding in the attention. The pos_bias_u/v vectors
//  ship as raw params (no `.weight`); the loader renames them to `.weight` so we can hold them
//  as Embeddings. The `pos_enc.pe` buffer is computed deterministically (not loaded).
//

import Foundation
import MLX
import MLXNN

// MARK: - Relative positional encoding

final class EspnetRelPositionalEncoding: Module {
    let dModel: Int
    let xscale: Float
    var pe: MLXArray

    init(_ dModel: Int, maxLen: Int = 5000) {
        self.dModel = dModel
        self.xscale = sqrt(Float(dModel))
        self.pe = EspnetRelPositionalEncoding.buildPE(dModel: dModel, size: maxLen)
        super.init()
    }

    static func buildPE(dModel: Int, size: Int) -> MLXArray {
        let half = dModel / 2
        let position = MLXArray((0 ..< size).map { Float($0) }).expandedDimensions(axis: -1)  // [size,1]
        let divTerm = exp(MLXArray((0 ..< half).map { Float($0) }) * Float(-log(10000.0) / Double(dModel)))
        let arg = position * divTerm                                   // [size, half]
        let pos = concatenated([sin(arg).expandedDimensions(axis: -1), cos(arg).expandedDimensions(axis: -1)], axis: -1)
            .reshaped([size, dModel])
        let nArg = -arg
        let neg = concatenated([sin(nArg).expandedDimensions(axis: -1), cos(nArg).expandedDimensions(axis: -1)], axis: -1)
            .reshaped([size, dModel])
        // pos reversed along axis 0; neg[1:]
        let revIdx = MLXArray((0 ..< size).reversed().map { Int32($0) })
        let posFlip = take(pos, revIdx, axis: 0)
        let negTail = neg[1..., 0...]
        let full = concatenated([posFlip, negTail], axis: 0)           // [2*size-1, dModel]
        return full.expandedDimensions(axis: 0)                        // [1, 2*size-1, dModel]
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let T = x.dim(1)
        let center = pe.dim(1) / 2
        let posEmb = pe[0..., (center - T + 1)..<(center + T), 0...]
        return (x * xscale, posEmb)
    }
}

final class LinearInput: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pos_enc") var posEnc: EspnetRelPositionalEncoding
    init(_ inSize: Int, _ outSize: Int) {
        _linear.wrappedValue = Linear(inSize, outSize)
        _norm.wrappedValue = LayerNorm(dimensions: outSize, eps: 1e-5)
        _posEnc.wrappedValue = EspnetRelPositionalEncoding(outSize)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        var h = norm(linear(x))
        let (scaled, posEmb) = posEnc(h)
        h = scaled
        return (h, posEmb, mask)
    }
}

// MARK: - Relative-position multi-head attention

final class RelPositionMultiHeadedAttention: Module {
    let nHead: Int, dK: Int, scale: Float
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ModuleInfo(key: "pos_bias_u") var posBiasU: Embedding   // [nHead, dK], raw param renamed to .weight
    @ModuleInfo(key: "pos_bias_v") var posBiasV: Embedding

    init(_ nHead: Int, _ nFeat: Int) {
        self.nHead = nHead
        self.dK = nFeat / nHead
        self.scale = pow(Float(dK), -0.5)
        _linearQ.wrappedValue = Linear(nFeat, nFeat)
        _linearK.wrappedValue = Linear(nFeat, nFeat, bias: true)
        _linearV.wrappedValue = Linear(nFeat, nFeat)
        _linearOut.wrappedValue = Linear(nFeat, nFeat)
        _linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        _posBiasU.wrappedValue = Embedding(embeddingCount: nHead, dimensions: dK)
        _posBiasV.wrappedValue = Embedding(embeddingCount: nHead, dimensions: dK)
        super.init()
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        // x: (B, nHead, T1, 2*T1-1)
        let B = x.dim(0), nh = x.dim(1), T1 = x.dim(2), T2 = x.dim(3)
        let zeroPad = MLXArray.zeros([B, nh, T1, 1])
        var xp = concatenated([zeroPad, x], axis: -1)              // [B, nh, T1, T2+1]
        xp = xp.reshaped([B, nh, T2 + 1, T1])
        xp = xp[0..., 0..., 1..., 0...]                           // drop first row
        xp = xp.reshaped([B, nh, T1, T2])
        return xp[0..., 0..., 0..., 0..<(T2 / 2 + 1)]
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, posEmb: MLXArray?) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), D = x.dim(2)
        let hs: [Int] = [B, T, nHead, dK]
        let q = linearQ(x).reshaped(hs)
        let k = linearK(x).reshaped(hs).transposed(0, 2, 1, 3)     // [B,nh,T,dk]
        let v = linearV(x).reshaped(hs).transposed(0, 2, 1, 3)

        let qWithU = (q + posBiasU.weight).transposed(0, 2, 1, 3)  // [B,nh,T,dk]
        var matrixAc = matmul(qWithU, k.transposed(0, 1, 3, 2))    // [B,nh,T,T]

        var scores: MLXArray
        if let posEmb {
            let Tpos = posEmb.dim(1)
            let p = linearPos(posEmb).reshaped([1, Tpos, nHead, dK]).transposed(0, 2, 1, 3)  // [1,nh,2T-1,dk]
            let qWithV = (q + posBiasV.weight).transposed(0, 2, 1, 3)
            var matrixBd = matmul(qWithV, p.transposed(0, 1, 3, 2))  // [B,nh,T,2T-1]
            if matrixAc.dim(3) != matrixBd.dim(3) { matrixBd = relShift(matrixBd) }
            scores = (matrixAc + matrixBd) * scale
            _ = matrixAc
        } else {
            scores = matrixAc * scale
        }

        if let mask, mask.dim(-1) == T {  // mask: (B, T)
            scores = MLX.where(mask.expandedDimensions(axis: 1).expandedDimensions(axis: 2) .> 0,
                               scores, MLXArray(-Float.infinity))
        }
        let attn = softmax(scores, axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped([B, T, D])
        return linearOut(out)
    }
}

final class PositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear
    init(_ dModel: Int, _ dInner: Int) {
        _w1.wrappedValue = Linear(dModel, dInner)
        _w2.wrappedValue = Linear(dInner, dModel)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x))) }
}

final class ConformerEncoderLayer: Module {
    @ModuleInfo(key: "norm_mha") var normMha: LayerNorm
    var selfAttn: RelPositionMultiHeadedAttention
    @ModuleInfo(key: "norm_ff") var normFf: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: PositionwiseFeedForward
    init(_ size: Int, _ nHead: Int, _ dInner: Int) {
        _normMha.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        self.selfAttn = RelPositionMultiHeadedAttention(nHead, size)
        _normFf.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        _feedForward.wrappedValue = PositionwiseFeedForward(size, dInner)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, posEmb: MLXArray?) -> MLXArray {
        var residual = x
        var h = normMha(x)
        residual = residual + selfAttn(h, mask: mask, posEmb: posEmb)
        h = normFf(residual)
        return residual + feedForward(h)
    }
}

final class PreLookaheadLayer: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    let preLookaheadLen: Int
    init(_ channels: Int, _ preLookaheadLen: Int = 3) {
        self.preLookaheadLen = preLookaheadLen
        _conv1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                     kernelSize: preLookaheadLen + 1, padding: 0)
        _conv2.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                     kernelSize: 3, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = padded(x, widths: [[0, 0], [0, preLookaheadLen], [0, 0]])
        out = leakyRelu(conv1(out), negativeSlope: 0.1)
        out = padded(out, widths: [[0, 0], [2, 0], [0, 0]])
        out = conv2(out)
        return out + x
    }
}

final class Upsample1DEncoder: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let stride: Int
    init(_ channels: Int, stride: Int = 2) {
        self.stride = stride
        _conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                    kernelSize: stride * 2 + 1, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, xLens: MLXArray) -> (MLXArray, MLXArray) {
        var y = repeated(x, count: stride, axis: 1)
        y = padded(y, widths: [[0, 0], [stride * 2, 0], [0, 0]])
        y = conv(y)
        return (y, xLens * stride)
    }
}

final class UpsampleConformerEncoder: Module {
    @ModuleInfo(key: "embed") var embed: LinearInput
    @ModuleInfo(key: "pre_lookahead_layer") var preLookahead: PreLookaheadLayer
    @ModuleInfo(key: "encoders") var encoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "up_layer") var upLayer: Upsample1DEncoder
    @ModuleInfo(key: "up_embed") var upEmbed: LinearInput
    @ModuleInfo(key: "up_encoders") var upEncoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    init(inputSize: Int, outputSize: Int, attentionHeads: Int, linearUnits: Int, numBlocks: Int) {
        _embed.wrappedValue = LinearInput(inputSize, outputSize)
        _preLookahead.wrappedValue = PreLookaheadLayer(outputSize, 3)
        _encoders.wrappedValue = (0..<numBlocks).map { _ in ConformerEncoderLayer(outputSize, attentionHeads, linearUnits) }
        _upLayer.wrappedValue = Upsample1DEncoder(outputSize, stride: 2)
        _upEmbed.wrappedValue = LinearInput(inputSize, outputSize)
        _upEncoders.wrappedValue = (0..<4).map { _ in ConformerEncoderLayer(outputSize, attentionHeads, linearUnits) }
        _afterNorm.wrappedValue = LayerNorm(dimensions: outputSize, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ xs: MLXArray, xsLens: MLXArray) -> (MLXArray, MLXArray) {
        let B = xs.dim(0), T = xs.dim(1)
        var mask = (MLXArray((0 ..< T).map { Int32($0) }) .< xsLens.expandedDimensions(axis: -1))   // (B, T)
        mask = mask.expandedDimensions(axis: 1)                                   // (B, 1, T)

        var (h, posEmb, _) = embed(xs, mask: mask)
        h = preLookahead(h)
        var mask1d = mask[0..., 0..<1, 0...].squeezed(axis: 1)                   // (B, T)
        for layer in encoders { h = layer(h, mask: mask1d, posEmb: posEmb) }

        var (hUp, lensUp) = upLayer(h, xLens: xsLens)
        let T2 = hUp.dim(1)
        mask = (MLXArray((0 ..< T2).map { Int32($0) }) .< lensUp.expandedDimensions(axis: -1)).expandedDimensions(axis: 1)
        let (h2, posEmb2, _) = upEmbed(hUp, mask: mask)
        hUp = h2
        mask1d = mask[0..., 0..<1, 0...].squeezed(axis: 1)
        for layer in upEncoders { hUp = layer(hUp, mask: mask1d, posEmb: posEmb2) }
        hUp = afterNorm(hUp)
        return (hUp, mask)
    }
}
