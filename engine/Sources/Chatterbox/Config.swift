//
//  Config.swift — Codable view of Chatterbox-Turbo's config.json
//

import Foundation

extension JSONDecoder {
    /// Shared decoder: Chatterbox's config.json uses snake_case keys.
    static var configDecoder: JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }
}

public struct ChatterboxQuantization: Codable {
    public let groupSize: Int
    public let bits: Int
    /// "affine" (we map to MLX's `.affine` QuantizationMode at load).
    public let mode: String
}

public struct GPT2Config: Codable {
    public let activationFunction: String
    public let nCtx: Int
    public let nEmbd: Int
    public let hiddenSize: Int
    public let nHead: Int
    public let nLayer: Int
    public let nPositions: Int
    public let vocabSize: Int
    public let layerNormEpsilon: Double
    public let attnPdrop: Double?
    public let embdPdrop: Double?
    public let residPdrop: Double?
}

public struct T3ConfigSection: Codable {
    public let startTextToken: Int
    public let stopTextToken: Int
    public let textTokensDictSize: Int
    public let maxTextTokens: Int?
    public let startSpeechToken: Int
    public let stopSpeechToken: Int
    public let speechTokensDictSize: Int
    public let maxSpeechTokens: Int
    public let llamaConfigName: String?
    /// null in the turbo config.
    public let inputPosEmb: Int?
    public let speechCondPromptLen: Int
    public let encoderType: String?
    public let speakerEmbedSize: Int
    public let usePerceiverResampler: Bool?
    public let emotionAdv: Bool?
}

public struct S3GenConfig: Codable {
    public let outputSampleRate: Int
    public let inputSampleRate: Int
    public let silenceToken: Int
    public let speechVocabSize: Int
    public let meanflow: Bool
    public let tokenEmbeddingDim: Int
    public let encoderAttentionHeads: Int
    public let encoderLinearUnits: Int
    public let encoderNumBlocks: Int
    public let encoderDropoutRate: Double?
    public let decoderInChannels: Int
    public let decoderOutChannels: Int
    public let decoderChannels: [Int]
    public let decoderAttentionHeadDim: Int
    public let decoderNBlocks: Int
    public let decoderNumMidBlocks: Int
    public let decoderNumHeads: Int
    public let cfmSigmaMin: Double
    public let cfmTScheduler: String
    public let cfmInferenceCfgRate: Double?
}

public struct VoiceEncoderConfig: Codable {
    public let numMels: Int
    public let sampleRate: Int
    public let speakerEmbedSize: Int
    public let veHiddenSize: Int
    public let nFft: Int
    public let hopSize: Int
    public let winSize: Int
    public let fmax: Int
    public let fmin: Int
    public let vePartialFrames: Int
    public let veFinalRelu: Bool?
}

public struct ChatterboxConfig: Codable {
    public let architecture: String
    public let modelType: String
    public let sampleRate: Int
    public let encCondLenSeconds: Int?
    public let decCondLenSeconds: Int?
    public let quantization: ChatterboxQuantization?
    public let gpt2: GPT2Config
    public let t3: T3ConfigSection
    public let s3gen: S3GenConfig
    public let voiceEncoder: VoiceEncoderConfig

    public static func load(at url: URL) throws -> ChatterboxConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder.configDecoder.decode(ChatterboxConfig.self, from: data)
    }
}
