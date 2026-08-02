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
import MLXLMCommon
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
        // Load the 4-bit model.safetensors (U32 packed + scales/biases) and quantize the Swift
        // module tree so update() can pour the packed weights into QuantizedLinear/Embedding —
        // this matches how mlx-audio's `load_model` (the working backend) holds the model.
        var weights = try ChatterboxLoader.loadWeights(files)
        let conds = try ChatterboxLoader.loadConds(files)
        // Structural renames so flat keys match the Swift module tree.
        weights = Dictionary(uniqueKeysWithValues: weights.map { (k, v) -> (String, MLXArray) in
            if k.hasSuffix(".pos_bias_u") || k.hasSuffix(".pos_bias_v") { return (k + ".weight", v) }
            if k.hasSuffix(".alpha") { return (k + ".weight", v.reshaped([v.dim(0), 1])) }
            if k.contains(".ff.net.0.") { return (k.replacingOccurrences(of: ".ff.net.0.", with: ".ff.net.gelu."), v) }
            if k.contains(".ff.net.1.") { return (k.replacingOccurrences(of: ".ff.net.1.", with: ".ff.net.out."), v) }
            return (k, v)
        })
        let quantPaths = ChatterboxLoader.quantizedPaths(weights)

        let model = ChatterboxTurbo(config: config, conds: conds)
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, module in
                if module is Linear { return quantPaths.contains(path) }
                if module is Embedding { return quantPaths.contains(path) }
                return false
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [])
        eval(model)

        model.tokenizer = try? await AutoTokenizer.from(modelFolder: files.dir)
        print("[Chatterbox] ✅ loaded (t3 \(config.gpt2.nLayer)L/\(config.gpt2.nEmbd)d, "
              + "tokenizer=\(model.tokenizer != nil ? "ok" : "MISSING"))")
        return model
    }

    /// Text → 24 kHz mono Float samples.
    public func generate(text: String) -> [Float] {
        // DEBUG ISOLATION: feed a precomputed embeds tensor (float32 [1, T, 1024]) straight
        // into the GPT2 forward to test the transformer independent of T3's embed prep.
        // CHATTERBOX_TEST_EMBEDS=<path>  (also set CHATTERBOX_DUMP=</dir>).
        if let p = ProcessInfo.processInfo.environment["CHATTERBOX_TEST_EMBEDS"],
           let data = try? Data(contentsOf: URL(fileURLWithPath: p)) {
            let n = data.count / 4
            let T = n / 1024
            let embeds = MLXArray(data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) })
                .reshaped([1, T, 1024])
            ChatterboxDump(embeds, "swift_fed_embeds")
            let cache: [KVCache] = (0..<t3.tfmr.config.nLayer).map { _ in KVCacheSimple() }
            let hidden = t3.tfmr(inputsEmbeds: embeds, cache: cache)
            let lastIdx = hidden.dim(1) - 1
            ChatterboxDump(hidden, "swift_t3_hidden_py")
            ChatterboxDump(t3.speechHead(hidden[0..., lastIdx, 0...]), "swift_t3_logits0_py")
            print("[Chatterbox] test-forward on external embeds dumped (T=\(T))")
            return []
        }

        // DEBUG ISOLATION: feed a precomputed mel (float32 [1,80,T] or [80,T]) into the vocoder
        // to test HiFTNet independent of S3. CHATTERBOX_TEST_MEL=<path>.
        if let p = ProcessInfo.processInfo.environment["CHATTERBOX_TEST_MEL"],
           let data = try? Data(contentsOf: URL(fileURLWithPath: p)) {
            let n = data.count / 4
            let T = n / 80
            let mel = MLXArray(data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) })
                .reshaped([1, 80, T])
            var wav = s3gen.mel2wav.generate(mel)
            wav = ChatterboxTurbo.applyTrimFade(wav)
            ChatterboxDump(MLXArray(wav), "swift_voc_wav")
            print("[Chatterbox] vocoder test on external mel dumped (T=\(T), samples=\(wav.count))")
            return wav
        }

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

        // Optional debug dump (CHATTERBOX_DUMP=/path) — tokens + mel + wav, to diff vs Python.
        if let d = ProcessInfo.processInfo.environment["CHATTERBOX_DUMP"] {
            let dir = URL(fileURLWithPath: d)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let toks = valid.map { Int($0) }
            try? JSONSerialization.data(withJSONObject: toks, options: [])
                .write(to: dir.appendingPathComponent("tokens.json"))
            let melFlat = mel[0].asArray(Float.self)
            try? melFlat.withUnsafeBufferPointer { Data(buffer: $0) }
                .write(to: dir.appendingPathComponent("swift_mel.bin"))
            try? Data("\(mel[0].dim(0)) \(mel[0].dim(1))".utf8)
                .write(to: dir.appendingPathComponent("swift_mel.shape"))
            try? wav.withUnsafeBufferPointer { Data(buffer: $0) }
                .write(to: dir.appendingPathComponent("swift_wav.bin"))
            print("[Chatterbox] dumped tokens/mel/wav → \(d)")
        }
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
