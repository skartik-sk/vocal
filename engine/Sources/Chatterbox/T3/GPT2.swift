//
//  GPT2.swift — GPT-2 backbone for T3 (text → speech-token LM).
//
//  Ported from chatterbox_turbo/models/t3/gpt2.py. A standard pre-LN GPT-2 decoder:
//  combined-QKV attention (c_attn), learned position embeddings (wpe), gelu-new MLP.
//

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXLMCommon

/// GPT-2 "new" GELU (tanh approximation).
func geluNew(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1.0 + tanh(sqrt(2.0 / Float.pi) * (x + 0.044715 * pow(x, 3.0))))
}

final class GPT2Attention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "c_attn") private(set) var cAttn: Linear
    @ModuleInfo(key: "c_proj") private(set) var cProj: Linear

    init(_ config: GPT2Config) {
        self.embedDim = config.nEmbd
        self.numHeads = config.nHead
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._cAttn.wrappedValue = Linear(embedDim, 3 * embedDim)
        self._cProj.wrappedValue = Linear(embedDim, embedDim)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache?) -> MLXArray {
        let B = hidden.dim(0), T = hidden.dim(1)
        let qkv = cAttn(hidden)                       // [B, T, 3C]
        let parts = split(qkv, parts: 3, axis: -1)
        var q = parts[0], k = parts[1], v = parts[2]
        let headShape: [Int] = [B, T, numHeads, headDim]
        q = q.reshaped(headShape).transposed(0, 2, 1, 3)   // [B, heads, T, headDim]
        k = k.reshaped(headShape).transposed(0, 2, 1, 3)
        v = v.reshaped(headShape).transposed(0, 2, 1, 3)

        // Manual attention matching the Python GPT2 exactly (explicit triu causal mask), rather
        // than MLXFast.sdpa `.causal` — the on-the-floor arithmetic is bit-for-bit the reference.
        let pastLen = cache?.offset ?? 0
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let qL = q.dim(2), kL = k.dim(2)
        let qIdx = (MLXArray((0..<qL).map { Int32($0) }) + Int32(pastLen)).reshaped([qL, 1])
        let kIdx = MLXArray((0..<kL).map { Int32($0) }).reshaped([1, kL])
        let causal = MLX.where(qIdx .>= kIdx, MLXArray(0.0), MLXArray(-Float.infinity))
        var attn = matmul(q, k.transposed(0, 1, 3, 2)) * scale   // [B, heads, qL, kL]
        attn = attn + causal
        attn = softmax(attn, axis: -1)
        let out = matmul(attn, v)                               // [B, heads, qL, headDim]
        return cProj(out.transposed(0, 2, 1, 3).reshaped([B, T, embedDim]))
    }
}

final class GPT2MLP: Module {
    @ModuleInfo(key: "c_fc") private(set) var cFc: Linear
    @ModuleInfo(key: "c_proj") private(set) var cProj: Linear

    init(_ config: GPT2Config) {
        let inner = 4 * config.nEmbd
        self._cFc.wrappedValue = Linear(config.nEmbd, inner)
        self._cProj.wrappedValue = Linear(inner, config.nEmbd)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { cProj(geluNew(cFc(x))) }
}

final class GPT2Block: Module {
    @ModuleInfo(key: "ln_1") private(set) var ln1: LayerNorm
    @ModuleInfo(key: "attn") private(set) var attn: GPT2Attention
    @ModuleInfo(key: "ln_2") private(set) var ln2: LayerNorm
    @ModuleInfo(key: "mlp") private(set) var mlp: GPT2MLP

    init(_ config: GPT2Config) {
        self._ln1.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: Float(config.layerNormEpsilon))
        self._attn.wrappedValue = GPT2Attention(config)
        self._ln2.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: Float(config.layerNormEpsilon))
        self._mlp.wrappedValue = GPT2MLP(config)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, cache: KVCache?) -> MLXArray {
        var h = hidden
        h = h + attn(ln1(h), cache: cache)
        h = h + mlp(ln2(h))
        return h
    }
}

final class GPT2Model: Module {
    let config: GPT2Config
    @ModuleInfo(key: "wte") private(set) var wte: Embedding
    @ModuleInfo(key: "wpe") private(set) var wpe: Embedding
    @ModuleInfo(key: "h") private(set) var h: [GPT2Block]
    @ModuleInfo(key: "ln_f") private(set) var lnF: LayerNorm

    init(_ config: GPT2Config) {
        self.config = config
        self._wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbd)
        self._wpe.wrappedValue = Embedding(embeddingCount: config.nPositions, dimensions: config.nEmbd)
        self._h.wrappedValue = (0..<config.nLayer).map { _ in GPT2Block(config) }
        self._lnF.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: Float(config.layerNormEpsilon))
        super.init()
    }

    func callAsFunction(inputsEmbeds: MLXArray, cache: [KVCache]) -> MLXArray {
        var hidden = inputsEmbeds
        let T = hidden.dim(1)
        let pastLength = cache.first?.offset ?? 0
        let positionIds = MLXArray((pastLength..<(pastLength + T)).map { Int32($0) })
        ChatterboxDump(positionIds, "swift_positionIds")
        ChatterboxDump(wpe.weight[0..<8, 0...], "swift_wpe_direct")   // rows 0-7 via slice
        let wpeOut = wpe(positionIds)
        ChatterboxDump(wpeOut, "swift_gpt2_wpe")
        hidden = hidden + wpeOut
        ChatterboxDump(hidden, "swift_gpt2_posemb")
        for i in 0..<h.count {
            hidden = h[i](hidden, cache: cache[i])
            if i == 0 { ChatterboxDump(hidden, "swift_gpt2_block0") }
        }
        return lnF(hidden)
    }
}
