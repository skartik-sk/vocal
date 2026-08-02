//
//  MLHindiTests.swift — verify the multilingual T3 produces Hindi speech tokens.
//

import XCTest
import MLX
import MLXNN
import MLXLMCommon
@testable import Chatterbox

final class MLHindiTests: XCTestCase {
    // The downloaded chatterbox-4bit model (override via CHATTERBOX_ML_MODEL).
    private static let modelPath = ProcessInfo.processInfo.environment["CHATTERBOX_ML_MODEL"]
        ?? "/tmp/chatterbox-4bit"

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw XCTSkip("chatterbox-4bit not found at \(Self.modelPath)")
        }
    }

    func testLoadsAndTokenizesHindi() async throws {
        let model = try await ChatterboxML.fromPretrained(Self.modelPath)
        XCTAssertNotNil(model.tokenizer)
        let toks = model.tokenizer!.tokenize(text: "नमस्ते, मैं हिंदी में बोल रहा हूँ।", languageID: "hi")
        print("[ML] hindi text tokens: \(toks)")
        // Python reference: [722, 1706, 1712, 1720, 1740, 1702, 1734, 7, 2, ...]
        XCTAssertEqual(toks.prefix(3), [722, 1706, 1712], "tokenizer should match Python MTL")
    }

    func testGeneratesHindiSpeechTokens() async throws {
        print("[ML] starting test")
        fflush(stdout)
        let model = try await ChatterboxML.fromPretrained(Self.modelPath)
        print("[ML] loaded ok")
        fflush(stdout)
        guard let tok = model.tokenizer else { XCTFail("no tokenizer"); return }
        let ids = tok.tokenize(text: "नमस्ते, मैं हिंदी में बोल रहा हूँ।", languageID: "hi")
        print("[ML] tokenized: \(ids.prefix(8)) count=\(ids.count)")
        fflush(stdout)
        let textTokens = MLXArray(ids.map { Int32($0) }).reshaped([1, -1])
        print("[ML] textTokens shape: \(textTokens.shape)")
        fflush(stdout)
        print("[ML] conds shapes: spk=\(model.conds.t3SpeakerEmb.shape) emo=\(model.conds.t3EmotionAdv.shape) prompt=\(model.conds.t3CondPromptSpeechTokens.shape)")
        fflush(stdout)
        let cond = T3MLCond(speakerEmb: model.conds.t3SpeakerEmb,
                            emotionAdv: model.conds.t3EmotionAdv,
                            condPromptSpeechTokens: model.conds.t3CondPromptSpeechTokens)
        print("[ML] building cond emb...")
        fflush(stdout)
        let condEmb = model.t3.prepareConditioning(cond)
        print("[ML] condEmb shape: \(condEmb.shape)")
        fflush(stdout)
        print("[ML] text emb shape: \(model.t3.textEmb(textTokens).shape)")
        fflush(stdout)
        // Test the Llama backbone alone
        let embeds = model.t3.prepareInputEmbeds(
            cond: cond, textTokens: textTokens, speechTokens: MLXArray([Int32(6561)]).reshaped([1, 1]),
            cfgWeight: 0).0
        print("[ML] input embeds shape: \(embeds.shape)")
        fflush(stdout)
        let cache: [KVCache] = (0 ..< model.t3.config.hiddenLayers).map { _ in KVCacheSimple() }
        let hidden = model.t3.tfmr(embeds, cache: cache)
        print("[ML] hidden shape: \(hidden.shape)")
        fflush(stdout)
        // test speechHead alone
        let sh = model.t3.speechHead as! QuantizedLinear
        print("[ML] speechHead groupSize=\(sh.groupSize) bits=\(sh.bits) weight=\(sh.weight.shape) scales=\(sh.scales.shape)")
        fflush(stdout)
        let lg = sh(hidden[0..., 69, 0...])
        print("[ML] logits shape: \(lg.shape)")
        fflush(stdout)
        // Verify embeddings dequantized correctly vs Python
        let e0 = model.t3.textEmb(MLXArray([Int32(722)]).reshaped([1, 1]))
        let e0arr = e0.asArray(Float.self)
        print("[ML] textEmb[722][0..5]: \(Array(e0arr.prefix(5)))")
        fflush(stdout)
        let toks = model.t3.inference(cond: cond, textTokens: textTokens,
                                      maxNewTokens: 60, temperature: 0.8)
        let flat = toks.asArray(Int32.self)
        print("[ML] hindi speech tokens: \(Array(flat.prefix(60))) count=\(flat.count)")
        // Python reference: [6561, 3677, 6486, 1960, 3913, ...] — starts with BOS
        XCTAssertGreaterThan(flat.count, 10)
        XCTAssertEqual(Int(flat[0]), 6561)
        // Should not be degenerate (all zeros / one repeated token)
        let unique = Set(flat.map { Int($0) })
        XCTAssertGreaterThan(unique.count, 10, "output should be diverse speech tokens")
    }
}
