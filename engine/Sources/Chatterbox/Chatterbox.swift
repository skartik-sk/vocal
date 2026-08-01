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
    public var tokenizer: Tokenizer?

    init(config: ChatterboxConfig, conds: Conds) {
        self.config = config
        self.conds = conds
        self._t3.wrappedValue = T3(gpt2: config.gpt2, hp: config.t3)
        super.init()
    }

    /// Load model weights + the baked `conds.safetensors` default voice from a model directory.
    public static func fromPretrained(_ modelPath: String) async throws -> ChatterboxTurbo {
        let files = ModelFiles(modelPath)
        let config = try ChatterboxConfig.load(at: files.configURL)
        let weights = try ChatterboxLoader.loadWeights(files)
        let conds = try ChatterboxLoader.loadConds(files)
        let quantPaths = ChatterboxLoader.quantizedPaths(weights)

        let model = ChatterboxTurbo(config: config, conds: conds)
        // Quantize linears (all) + only the embeddings that ship quantized on disk.
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, module in
                if module is Embedding { return quantPaths.contains(path) }
                return true
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
        GPU.clearCache()

        // M3+ wire S3 (tokens→mel) + HiFTNet (mel→wav) here. Until then, a short tone keeps the
        // worker audible and proves the full T3 forward pass ran.
        return ChatterboxTurbo.stubTone()
    }

    /// A short 440 Hz tone, used until the vocoder is wired (M4).
    public static func stubTone() -> [Float] {
        let sr = sampleRate
        let n = Int(0.4 * Double(sr))
        return (0..<n).map { i in Float(sin(2.0 * .pi * 440.0 * Double(i) / Double(sr)) * 0.2) }
    }
}
