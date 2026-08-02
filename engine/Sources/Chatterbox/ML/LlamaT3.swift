//
//  LlamaT3.swift — Llama backbone for the multilingual Chatterbox T3
//  (text → speech-token LM). Ported from mlx-audio chatterbox/t3/t3.py.
//
//  Multilingual (non-turbo) model from mlx-community/chatterbox-4bit:
//  30-layer Llama, hidden 1024, 16 heads (head_dim 64), SwiGLU MLP
//  (intermediate 4096), RoPE (llama3 scaling), RMSNorm. Accepts pre-built
//  input embeddings (the T3 feeds [cond | text | speech] embeds), so the
//  token-embedding table is bypassed.
//

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXLMCommon

// MARK: - Config

/// T3 hyperparameters from mlx-audio chatterbox/config.py (T3Config.multilingual).
public struct LlamaT3Config {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDimensions: Int
    public let kvHeads: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let mlpBias: Bool

    // T3-specific
    public let textTokensDictSize: Int
    public let speechTokensDictSize: Int
    public let startSpeechToken: Int
    public let stopSpeechToken: Int
    public let speakerEmbedSize: Int
    public let speechCondPromptLen: Int
    public let quantization: ChatterboxQuantization?

    public init(
        hiddenSize: Int = 1024, hiddenLayers: Int = 30, intermediateSize: Int = 4096,
        attentionHeads: Int = 16, headDimensions: Int = 64, kvHeads: Int = 16,
        rmsNormEps: Float = 1e-5, ropeTheta: Float = 500000,
        maxPositionEmbeddings: Int = 131072, attentionBias: Bool = false,
        mlpBias: Bool = false, textTokensDictSize: Int = 2454,
        speechTokensDictSize: Int = 8194, startSpeechToken: Int = 6561,
        stopSpeechToken: Int = 6562, speakerEmbedSize: Int = 256,
        speechCondPromptLen: Int = 150
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.kvHeads = kvHeads
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = [
            "factor": .float(8.0),
            "high_freq_factor": .float(4.0),
            "low_freq_factor": .float(1.0),
            "original_max_position_embeddings": .float(8192),
            "type": .string("llama3"),
        ]
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.textTokensDictSize = textTokensDictSize
        self.speechTokensDictSize = speechTokensDictSize
        self.startSpeechToken = startSpeechToken
        self.stopSpeechToken = stopSpeechToken
        self.speakerEmbedSize = speakerEmbedSize
        self.speechCondPromptLen = speechCondPromptLen
        self.quantization = ChatterboxQuantization(groupSize: 64, bits: 4, mode: "affine")
    }
}

// MARK: - RoPE

/// Llama3 rotary position embedding with dynamic base-frequency scaling.
final class LlamaRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    var base: Float?
    let scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?
    var freqs: MLXArray?

    init(
        dims: Int, maxPositionEmbeddings: Int?, traditional: Bool = false,
        base: Float = 10000, scale: Float = 1.0, ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings ?? 2048
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
        super.init()
        computeFreqs()
    }

    private func computeFreqs() {
        if ropeType != "llama3" {
            freqs = nil
            return
        }
        guard let ropeScaling,
            case .float(let factor) = ropeScaling["factor"],
            case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
            case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
            case .float(let oldContextLen) = ropeScaling["original_max_position_embeddings"]
                ?? .float(8192),
            let base
        else {
            freqs = nil
            return
        }
        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor
        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies
        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen), frequencies * factor, frequencies)
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen))
        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)
        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        self.base = nil
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dims, traditional: traditional, base: base,
            scale: scale, offset: offset, freqs: freqs)
    }
}

// MARK: - Attention / MLP / Block

/// Llama attention: separate q/k/v, RoPE, KV cache, output projection.
final class LlamaAttention: Module {
    let args: LlamaT3Config
    let scale: Float
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    let rope: LlamaRoPE

    init(_ config: LlamaT3Config) {
        self.args = config
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDimensions
        self.scale = pow(Float(headDim), -0.5)
        _wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)
        self.rope = LlamaRoPE(
            dims: headDim, maxPositionEmbeddings: config.maxPositionEmbeddings,
            traditional: false, base: config.ropeTheta, scale: 1.0,
            ropeType: "llama3", ropeScaling: config.ropeScaling)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        var values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: .none)
        return wo(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Llama MLP (SwiGLU).
final class LlamaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    init(_ config: LlamaT3Config) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Llama decoder block.
final class LlamaTransformerBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputLN: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: LlamaAttention
    @ModuleInfo(key: "post_attention_layernorm") var postLN: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: LlamaMLP
    init(_ config: LlamaT3Config) {
        _inputLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attention.wrappedValue = LlamaAttention(config)
        _postLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = LlamaMLP(config)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        var h = attention(inputLN(x), cache: cache)
        let y = x + h
        h = mlp(postLN(y))
        return y + h
    }
}

// MARK: - Backbone

/// Llama decoder over pre-built input embeddings (no token embedding table).
final class LlamaT3Backbone: Module {
    let config: LlamaT3Config
    @ModuleInfo(key: "layers") var layers: [LlamaTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: LlamaT3Config) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in LlamaTransformerBlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ embeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeds
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: cache?[i])
        }
        return norm(h)
    }
}
