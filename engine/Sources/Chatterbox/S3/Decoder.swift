//
//  Decoder.swift — ConditionalDecoder (the mean-flow CFM velocity estimator).
//
//  Ported from chatterbox_turbo/models/s3gen/decoder.py. A 1D UNet: 1 DownBlock → 12 MidBlocks
//  → 1 UpBlock, each Resnet(CausalConv+LayerNorm+Mish) + transformer blocks. Data flows in
//  (B, C, T) and transposes to (B, T, C) for conv / attention / norm, matching the Python.
//  MLXNN Conv1d weight is [O, K, I] — the stored weights are already in that layout.
//

import Foundation
import MLX
import MLXNN

/// Helper for the polymorphic downsample/upsample modules (CausalConv1d vs Downsample1D).
private protocol _Applies1D { func _apply(_ x: MLXArray) -> MLXArray }
extension CausalConv1d: _Applies1D { func _apply(_ x: MLXArray) -> MLXArray { self(x) } }
extension Downsample1D: _Applies1D { func _apply(_ x: MLXArray) -> MLXArray { self(x) } }
extension Upsample1D: _Applies1D { func _apply(_ x: MLXArray) -> MLXArray { self(x) } }

/// Take every k-th element along an axis (Python `x[..., ::k]`).
func strided(_ x: MLXArray, axis: Int, by k: Int) -> MLXArray {
    let n = x.dim(axis)
    let idx = MLXArray(Swift.stride(from: 0, to: n, by: k).map { Int32($0) })
    return take(x, idx, axis: axis)
}

// MARK: - Conv wrappers

/// Conv1d that accepts (B, C, T). weight path: `.conv.conv`.
final class Conv1dPT: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    init(_ inCh: Int, _ outCh: Int, kernel k: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inCh, outputChannels: outCh, kernelSize: k,
            stride: stride, padding: padding, dilation: dilation)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x.transposed(0, 2, 1)).transposed(0, 2, 1) }
}

/// ConvTranspose1d that accepts (B, C, T). weight path: `.conv.conv`.
final class ConvTranspose1dPT: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d
    init(_ inCh: Int, _ outCh: Int, kernel k: Int, stride: Int = 1, padding: Int = 0) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inCh, outputChannels: outCh, kernelSize: k, stride: stride, padding: padding)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x.transposed(0, 2, 1)).transposed(0, 2, 1) }
}

/// Causal 1D conv: left-pad time, then a no-pad conv. weight path: `.conv.conv`.
final class CausalConv1d: Module {
    let causalPadding: Int
    @ModuleInfo(key: "conv") var conv: Conv1dPT
    init(_ inCh: Int, _ outCh: Int, kernel k: Int, dilation: Int = 1) {
        self.causalPadding = (k - 1) * dilation
        _conv.wrappedValue = Conv1dPT(inCh, outCh, kernel: k, padding: 0, dilation: dilation)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if causalPadding > 0 { y = padded(y, widths: [[0, 0], [0, 0], [causalPadding, 0]]) }
        return conv(y)
    }
}

// MARK: - Time embedding

func sinusoidalPosEmb(_ t: MLXArray, _ dim: Int, scale: Float = 1000) -> MLXArray {
    let half = dim / 2
    let emb = exp(MLXArray((0 ..< half).map { Float($0) }) * Float(-log(10000.0) / Double(half - 1)))
    let arg = (scale * t).expandedDimensions(axis: -1) * emb
    return concatenated([sin(arg), cos(arg)], axis: -1)
}

final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var lin1: Linear
    @ModuleInfo(key: "linear_2") var lin2: Linear
    init(_ inCh: Int, _ dim: Int) {
        _lin1.wrappedValue = Linear(inCh, dim)
        _lin2.wrappedValue = Linear(dim, dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { lin2(silu(lin1(x))) }
}

// MARK: - Blocks

/// Causal block: CausalConv -> LayerNorm -> Mish. weight: `.block.0` (conv), `.block.1` (norm).
final class CausalBlock1D: Module {
    @ModuleInfo(key: "block") var block: [Module]
    init(_ dim: Int, _ dimOut: Int) {
        _block.wrappedValue = [CausalConv1d(dim, dimOut, kernel: 3), LayerNorm(dimensions: dimOut, eps: 1e-5)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x * mask
        h = (block[0] as! CausalConv1d)(h).transposed(0, 2, 1)
        h = (block[1] as! LayerNorm)(h).transposed(0, 2, 1)
        return mish(h) * mask
    }
}

final class ResnetBlock1D: Module {
    @ModuleInfo(key: "mlp") var mlp: [Linear]    // PyTorch mlp.1 -> index 0 here
    let block1: CausalBlock1D
    let block2: CausalBlock1D
    @ModuleInfo(key: "res_conv") var resConv: Conv1dPT
    init(_ dim: Int, _ dimOut: Int, _ timeDim: Int) {
        _mlp.wrappedValue = [Linear(timeDim, dimOut)]
        self.block1 = CausalBlock1D(dim, dimOut)
        self.block2 = CausalBlock1D(dimOut, dimOut)
        _resConv.wrappedValue = Conv1dPT(dim, dimOut, kernel: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = block1(x, mask: mask)
        h = h + mlp[0](mish(timeEmb)).expandedDimensions(axis: -1)
        h = block2(h, mask: mask)
        return h + resConv(x * mask)
    }
}

final class Downsample1D: Module {
    let conv: Conv1dPT
    init(_ dim: Int) { self.conv = Conv1dPT(dim, dim, kernel: 3, stride: 2, padding: 1); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class Upsample1D: Module {
    let conv: ConvTranspose1dPT
    init(_ dim: Int) { self.conv = ConvTranspose1dPT(dim, dim, kernel: 4, stride: 2, padding: 1); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class SelfAttention1D: Module {
    let numHeads: Int, headDim: Int, scale: Float
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    init(_ dim: Int, numHeads: Int, headDim: Int) {
        self.numHeads = numHeads; self.headDim = headDim; self.scale = pow(Float(headDim), -0.5)
        let inner = numHeads * headDim
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _toOut.wrappedValue = [Linear(inner, dim)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let hs: [Int] = [B, T, numHeads, headDim]
        let q = toQ(x).reshaped(hs).transposed(0, 2, 1, 3)
        let k = toK(x).reshaped(hs).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped(hs).transposed(0, 2, 1, 3)
        var attn = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let mask {  // mask: (B, T)
            attn = MLX.where(mask.expandedDimensions(axis: 1).expandedDimensions(axis: 2) .> 0, attn, MLXArray(-1e9))
        }
        attn = softmax(attn, axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped([B, T, -1])
        return toOut[0](out)
    }
}

/// GELU + linear (diffusers GELU). weight: `.proj`.
final class GELU: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ dimIn: Int, _ dimOut: Int) { _proj.wrappedValue = Linear(dimIn, dimOut); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(proj(x)) }
}

final class FeedForward: Module {
    @ModuleInfo(key: "net") private(set) var net: FFNetSeq
    init(_ dim: Int, _ mult: Int) { _net.wrappedValue = FFNetSeq(dim, dim * mult); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { net(x) }
}

/// FeedForward body as a MAP (not a [Module] array) — MLXNN unflattens integer `net.0/1` keys as
/// an array, which conflicts with a heterogeneous element structure. Non-integer keys `gelu`/
/// `out` unflatten as a map; the loader renames `net.0`→`net.gelu`, `net.1`→`net.out` to match.
final class FFNetSeq: Module {
    @ModuleInfo(key: "gelu") private(set) var g: GELU
    @ModuleInfo(key: "out") private(set) var l: Linear
    init(_ dim: Int, _ inner: Int) {
        _g.wrappedValue = GELU(dim, inner)
        _l.wrappedValue = Linear(inner, dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { l(g(x)) }
}

final class TransformerBlock: Module {
    @ModuleInfo(key: "attn1") var attn1: SelfAttention1D
    @ModuleInfo(key: "ff") var ff: FeedForward
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    init(_ dim: Int, numHeads: Int, headDim: Int) {
        _attn1.wrappedValue = SelfAttention1D(dim, numHeads: numHeads, headDim: headDim)
        _ff.wrappedValue = FeedForward(dim, 4)
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
        _norm3.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var y = x + attn1(norm1(x), mask: mask)
        y = y + ff(norm3(y))
        return y
    }
}

// MARK: - Down / Mid / Up

final class DownBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]
    @ModuleInfo(key: "downsample") var downsample: Module
    init(inputCh: Int, outputCh: Int, timeDim: Int, nBlocks: Int, numHeads: Int, headDim: Int, isLast: Bool) {
        _resnet.wrappedValue = ResnetBlock1D(inputCh, outputCh, timeDim)
        _transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in TransformerBlock(outputCh, numHeads: numHeads, headDim: headDim) }
        _downsample.wrappedValue = isLast ? CausalConv1d(outputCh, outputCh, kernel: 3) : Downsample1D(outputCh)
        super.init()
    }
}

final class MidBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]
    init(_ ch: Int, _ timeDim: Int, nBlocks: Int, numHeads: Int, headDim: Int) {
        _resnet.wrappedValue = ResnetBlock1D(ch, ch, timeDim)
        _transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in TransformerBlock(ch, numHeads: numHeads, headDim: headDim) }
        super.init()
    }
}

final class UpBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]
    @ModuleInfo(key: "upsample") var upsample: Module
    init(inputCh: Int, outputCh: Int, timeDim: Int, nBlocks: Int, numHeads: Int, headDim: Int, isLast: Bool) {
        _resnet.wrappedValue = ResnetBlock1D(inputCh, outputCh, timeDim)
        _transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in TransformerBlock(outputCh, numHeads: numHeads, headDim: headDim) }
        _upsample.wrappedValue = isLast ? CausalConv1d(outputCh, outputCh, kernel: 3) : Upsample1D(outputCh)
        super.init()
    }
}

// MARK: - ConditionalDecoder

final class ConditionalDecoder: Module {
    let inChannels: Int
    let meanflow: Bool
    @ModuleInfo(key: "time_mlp") var timeMlp: TimestepEmbedding
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock]
    @ModuleInfo(key: "mid_blocks") var midBlocks: [MidBlock]
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock]
    @ModuleInfo(key: "final_block") var finalBlock: CausalBlock1D
    @ModuleInfo(key: "final_proj") var finalProj: Conv1dPT
    @ModuleInfo(key: "time_embed_mixer") var timeEmbedMixer: Linear?

    init(inChannels: Int = 320, outChannels: Int = 80, channels: [Int] = [256],
         attentionHeadDim: Int = 64, nBlocks: Int = 4, numMidBlocks: Int = 12,
         numHeads: Int = 8, meanflow: Bool = false) {
        self.inChannels = inChannels
        self.meanflow = meanflow
        let timeDim = channels[0] * 4
        _timeMlp.wrappedValue = TimestepEmbedding(inChannels, timeDim)

        var db = [DownBlock]()
        var outputCh = inChannels
        for (i, ch) in channels.enumerated() {
            let inputCh = outputCh; outputCh = ch
            db.append(DownBlock(inputCh: inputCh, outputCh: outputCh, timeDim: timeDim,
                                nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim,
                                isLast: i == channels.count - 1))
        }
        _downBlocks.wrappedValue = db

        _midBlocks.wrappedValue = (0..<numMidBlocks).map { _ in
            MidBlock(channels.last!, timeDim, nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim)
        }

        let channelsUp = Array(channels.reversed()) + [channels[0]]
        var ub = [UpBlock]()
        for i in 0..<(channelsUp.count - 1) {
            let inputCh = channelsUp[i] * 2
            ub.append(UpBlock(inputCh: inputCh, outputCh: channelsUp[i + 1], timeDim: timeDim,
                              nBlocks: nBlocks, numHeads: numHeads, headDim: attentionHeadDim,
                              isLast: i == channelsUp.count - 2))
        }
        _upBlocks.wrappedValue = ub

        _finalBlock.wrappedValue = CausalBlock1D(channelsUp.last!, channelsUp.last!)
        _finalProj.wrappedValue = Conv1dPT(channelsUp.last!, outChannels, kernel: 1)
        _timeEmbedMixer.wrappedValue = meanflow ? Linear(timeDim * 2, timeDim, bias: false) : nil
        super.init()
    }

    func callAsFunction(
        x: MLXArray, mask: MLXArray, mu: MLXArray, t: MLXArray,
        spks: MLXArray?, cond: MLXArray?, r: MLXArray?
    ) -> MLXArray {
        var tEmb = timeMlp(sinusoidalPosEmb(t, inChannels))
        if meanflow, let r {
            let rEmb = timeMlp(sinusoidalPosEmb(r, inChannels))
            tEmb = timeEmbedMixer!(concatenated([tEmb, rEmb], axis: -1))
        }

        var inputs = [x, mu]
        if let spks {
            inputs.append(broadcast(spks.expandedDimensions(axis: -1), to: [spks.dim(0), spks.dim(1), x.dim(2)]))
        }
        if let cond { inputs.append(cond) }
        var h = concatenated(inputs, axis: 1)   // (B, 320, T)

        // Down path
        var hiddens = [MLXArray]()
        var masks = [mask]
        for db in downBlocks {
            let maskDown = masks[masks.count - 1]
            h = db.resnet(h, mask: maskDown, timeEmb: tEmb)
            h = h.transposed(0, 2, 1)
            let maskT = maskDown[0..., 0..<1, 0...].squeezed(axis: 1)
            for tb in db.transformerBlocks { h = tb(h, mask: maskT) }
            h = h.transposed(0, 2, 1)
            hiddens.append(h)
            let downOp = db.downsample as! _Applies1D
            h = downOp._apply(h * maskDown)
            masks.append(strided(maskDown, axis: 2, by: 2))
        }
        masks.removeLast()
        let maskMid = masks[masks.count - 1]

        // Mid path
        for mb in midBlocks {
            h = mb.resnet(h, mask: maskMid, timeEmb: tEmb)
            h = h.transposed(0, 2, 1)
            let maskT = maskMid[0..., 0..<1, 0...].squeezed(axis: 1)
            for tb in mb.transformerBlocks { h = tb(h, mask: maskT) }
            h = h.transposed(0, 2, 1)
        }

        // Up path
        var maskUp = maskMid
        for ub in upBlocks {
            maskUp = masks.removeLast()
            let skip = hiddens.removeLast()
            h = h[0..., 0..., 0..<skip.dim(2)]
            h = concatenated([h, skip], axis: 1)
            h = ub.resnet(h, mask: maskUp, timeEmb: tEmb)
            h = h.transposed(0, 2, 1)
            let maskT = maskUp[0..., 0..<1, 0...].squeezed(axis: 1)
            for tb in ub.transformerBlocks { h = tb(h, mask: maskT) }
            h = h.transposed(0, 2, 1)
            let upOp = ub.upsample as! _Applies1D
            h = upOp._apply(h * maskUp)
        }

        // Final
        h = finalBlock(h, mask: maskUp)
        h = finalProj(h * maskUp)
        return h * mask
    }
}
