//
//  Chatterbox  — native Swift/MLX port of Chatterbox-Turbo (default voice)
//
//  Pure-Swift reimplementation (zero Python) ported from
//  `mlx_audio.tts.models.chatterbox_turbo`. Pipeline: text → GPT2 BPE → T3 (GPT-2 AR LM) →
//  speech tokens → [M3] S3 (Conformer + mean-flow CFM) → mel → [M4] HiFTNet vocoder → 24kHz wav.
//
//  MILESTONE STATUS:
//    M0/M1 done. M2 (now): T3 wired + weight loading + tokenizer; `generate` runs T3 and
//    reports the speech-token count. Real audio arrives at M3 (S3) + M4 (vocoder).
//

import Foundation
import MLX
import MLXNN
import Tokenizers

/// Top-level native Chatterbox-Turbo model.
public final class ChatterboxTurbo: Module {
    public static let sampleRate = 24000

    public let config: ChatterboxConfig
    public let conds: Conds
    @ModuleInfo(key: "t3") var t3: T3
    @ModuleInfo(key: "s3gen") var s3gen: S3Gen
    public var tokenizer: Tokenizer?

    init(config: ChatterboxConfig, conds: Conds) {
        self.config = config
        self.conds = conds
        self._t3.wrappedValue = T3(gpt2: config.gpt2, hp: config.t3)
        self._s3gen.wrappedValue = S3Gen(meanflow: config.s3gen.meanflow)
        super.init()
    }

    /// Load model weights + the baked `conds.safetensors` default voice from a model directory.
    public static func fromPretrained(_ modelPath: String) async throws -> ChatterboxTurbo {
        let files = ModelFiles(modelPath)
        let config = try ChatterboxConfig.load(at: files.configURL)
        var weights = try ChatterboxLoader.loadWeights(files)
        let conds = try ChatterboxLoader.loadConds(files)
        // pos_bias_u/v ship as raw params (no `.weight` suffix); rename so the Embeddings that
        // hold them match. (pos_enc.pe is computed locally, so it stays an ignored extra key.)
        weights = Dictionary(uniqueKeysWithValues: weights.map { (k, v) -> (String, MLXArray) in
            if k.hasSuffix(".pos_bias_u") || k.hasSuffix(".pos_bias_v") { return (k + ".weight", v) }
            // Snake.alpha ships as a raw [channels] param; hold as Embedding(channels,1).weight.
            if k.hasSuffix(".alpha") { return (k + ".weight", v.reshaped([v.dim(0), 1])) }
            // FeedForward.net ships as integer-indexed (net.0/net.1); remap to map keys so the
            // heterogeneous structure loads (see FFNetSeq).
            if k.contains(".ff.net.0.") { return (k.replacingOccurrences(of: ".ff.net.0.", with: ".ff.net.gelu."), v) }
            if k.contains(".ff.net.1.") { return (k.replacingOccurrences(of: ".ff.net.1.", with: ".ff.net.out."), v) }
            return (k, v)
        })
        // Build the quantized-path set AFTER the rename so module paths match (e.g. ff.net.gelu).
        let quantPaths = ChatterboxLoader.quantizedPaths(weights)

        let model = ChatterboxTurbo(config: config, conds: conds)
        // Quantize only the linears/embeddings that ship quantized on disk. m_source.l_linear
        // (Linear 9→1) ships float (9 not divisible by group 64) and must stay float.
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, module in
                if module is Linear { return quantPaths.contains(path) }
                if module is Embedding { return quantPaths.contains(path) }
                return false
            }
        }
        // Pour weights in by dotted path; t3.* matches the `t3` submodule, s3gen.*/ve.* are
        // ignored (not built yet). verify:[] = don't fail on extra/missing keys.
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [])
        eval(model)

        model.tokenizer = try? await AutoTokenizer.from(modelFolder: files.dir)
        print("[Chatterbox] ✅ loaded (t3 \(config.gpt2.nLayer)L/\(config.gpt2.nEmbd)d, "
              + "tokenizer=\(model.tokenizer != nil ? "ok" : "MISSING"))")
        return model
    }

    /// Text → 24 kHz mono Float samples.
    public func generate(text: String) -> [Float] {
        guard let tok = tokenizer else {
            print("[Chatterbox] ⚠️ no tokenizer; playing stub tone")
            return ChatterboxTurbo.stubTone()
        }

        // GPT-2 BPE text tokens (turbo uses raw ids — no start/stop text-token wrapping).
        let ids = tok.encode(text: text)
        let textTokens = MLXArray(ids.map { Int32($0) }).reshaped([1, -1])

        let cond = T3Cond(
            speakerEmb: conds.t3SpeakerEmb,
            condPromptSpeechTokens: conds.t3CondPromptSpeechTokens)

        let rawTokens = t3.inference(cond: cond, textTokens: textTokens)
        // Drop special/OOV tokens (>= 6561), append 3× silence (4299) — matches Python generate.
        let flat = rawTokens.reshaped([-1]).asArray(Int32.self)
        let valid = Array(flat.filter { $0 < Int32(6561) }
                          + [Int32(4299), Int32(4299), Int32(4299)])
        print("[Chatterbox] T3 → \(valid.count) speech tokens (first 20: \(valid.prefix(20)))")

        // S3: speech tokens → mel (mean-flow CFM), conditioned on the baked default voice.
        let ref = S3Ref(promptToken: conds.genPromptToken, promptTokenLen: conds.genPromptTokenLen,
                        promptFeat: conds.genPromptFeat, embedding: conds.genEmbedding)
        let mel = s3gen(MLXArray(valid).reshaped([1, -1]), ref: ref)
        eval(mel)
        let melv = mel.asArray(Float.self)
        let lo = melv.min() ?? 0, hi = melv.max() ?? 0
        print("[Chatterbox] S3 mel shape \(mel.shape) range[\(String(format: "%.2f", lo)), \(String(format: "%.2f", hi))] frames=\(mel.dim(2))")
        GPU.clearCache()

        // HiFTNet vocoder: mel → 24 kHz mono wav.
        var wav = s3gen.mel2wav.generate(mel)
        wav = ChatterboxTurbo.applyTrimFade(wav)
        print("[Chatterbox] vocoder → \(wav.count) samples "
              + "(\(String(format: "%.2f", Double(wav.count) / Double(ChatterboxTurbo.sampleRate))) s)")
        GPU.clearCache()
        return wav
    }

    /// Fade-in the first 20 ms to suppress the startup artifact (matches Python `trim_fade`).
    static func applyTrimFade(_ wav: [Float]) -> [Float] {
        let nTrim = sampleRate / 50          // 480 samples = 20 ms
        guard wav.count >= 2 * nTrim else { return wav }
        var out = wav
        for i in 0..<nTrim {
            out[i] = 0
            let fade = (cos(.pi * Double(i) / Double(nTrim - 1)) + 1) / 2   // 0 → 1
            out[nTrim + i] = wav[nTrim + i] * Float(fade)
        }
        return out
    }

    /// A short 440 Hz tone (kept for fallback / debugging).
    public static func stubTone() -> [Float] {
        let sr = sampleRate
        let n = Int(0.4 * Double(sr))
        return (0..<n).map { i in Float(sin(2.0 * .pi * 440.0 * Double(i) / Double(sr)) * 0.2) }
    }
}
