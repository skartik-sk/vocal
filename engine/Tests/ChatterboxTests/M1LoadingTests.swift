//
//  M1LoadingTests.swift — verify the loading primitives against the REAL 4-bit checkpoint.
//

import XCTest
import MLX
@testable import Chatterbox

final class M1LoadingTests: XCTestCase {
    // The cached chatterbox-turbo-4bit snapshot. Override via CHATTERBOX_MODEL_PATH.
    private static let modelPath = ProcessInfo.processInfo.environment["CHATTERBOX_MODEL_PATH"]
        ?? "~/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-4bit/snapshots/<snapshot>"

    private var files: ModelFiles { ModelFiles(Self.modelPath) }

    override func setUpWithError() throws {
        try super.setUpWithError()
        // Skip cleanly if the model isn't downloaded on this machine.
        guard FileManager.default.fileExists(atPath: Self.modelPath) else {
            throw XCTSkip("chatterbox-turbo-4bit not found at \(Self.modelPath)")
        }
    }

    func testConfigParses() throws {
        let cfg = try ChatterboxConfig.load(at: files.configURL)
        XCTAssertEqual(cfg.modelType, "chatterbox_turbo")
        XCTAssertEqual(cfg.sampleRate, 24000)
        // GPT-2 medium
        XCTAssertEqual(cfg.gpt2.nLayer, 24)
        XCTAssertEqual(cfg.gpt2.nEmbd, 1024)
        XCTAssertEqual(cfg.gpt2.nHead, 16)
        XCTAssertEqual(cfg.gpt2.vocabSize, 50276)
        // T3 token boundaries
        XCTAssertEqual(cfg.t3.startSpeechToken, 6561)
        XCTAssertEqual(cfg.t3.stopSpeechToken, 6562)
        XCTAssertEqual(cfg.t3.speechTokensDictSize, 6563)
        XCTAssertEqual(cfg.t3.speechCondPromptLen, 375)
        // S3 — mean-flow, the turbo fast path
        XCTAssertEqual(cfg.s3gen.meanflow, true)
        XCTAssertEqual(cfg.s3gen.speechVocabSize, 6561)
        XCTAssertEqual(cfg.s3gen.silenceToken, 4299)
        XCTAssertEqual(cfg.s3gen.decoderNumMidBlocks, 12)
        XCTAssertEqual(cfg.s3gen.tokenEmbeddingDim, 512)
        // 4-bit affine, group 64
        XCTAssertEqual(cfg.quantization?.bits, 4)
        XCTAssertEqual(cfg.quantization?.groupSize, 64)
        XCTAssertEqual(cfg.quantization?.mode, "affine")
    }

    func testWeightsLoad() throws {
        let w = try ChatterboxLoader.loadWeights(files)
        // The index reported 2750 tensors.
        XCTAssertGreaterThan(w.count, 2000, "expected ~2750 weight tensors")
        // Spot-check the three component prefixes.
        XCTAssertTrue(w.keys.contains("t3.tfmr.h.0.attn.c_attn.weight"), "missing t3 gpt2 weight")
        XCTAssertTrue(w.keys.contains("s3gen.input_embedding.weight"), "missing s3gen weight")
        XCTAssertTrue(w.keys.contains { $0.hasPrefix("ve.lstm") }, "missing ve weight")
    }

    func testQuantizationLayout() throws {
        let w = try ChatterboxLoader.loadWeights(files)

        // Convs are full float16 with NO quantization siblings (the convert script leaves
        // convs unquantized; only linears/embeddings are 4-bit).
        let conv = "s3gen.decoder.estimator.down_blocks.0.downsample.conv.conv"
        XCTAssertEqual(w[conv + ".weight"]?.dtype, .float16, "conv weight should be float16")
        XCTAssertEqual(w[conv + ".weight"]?.ndim, 3)
        XCTAssertFalse(w.keys.contains(conv + ".scales"), "convs must not be quantized")

        // Linears ARE quantized: packed uint32 weight + 2-D float16 scales + biases.
        let lin = "t3.tfmr.h.0.attn.c_attn"
        XCTAssertEqual(w[lin + ".weight"]?.dtype, .uint32, "linear weight should be packed uint32")
        XCTAssertEqual(w[lin + ".scales"]?.ndim, 2)
        XCTAssertNotNil(w[lin + ".biases"])

        // Embeddings are quantized too.
        let emb = "s3gen.input_embedding"
        XCTAssertEqual(w[emb + ".weight"]?.dtype, .uint32)

        // Every .scales in the checkpoint is 2-D (there are NO quantized convs anywhere).
        let nonFlat = w.keys.filter { $0.hasSuffix(".scales") && (w[$0]?.ndim ?? 0) != 2 }
        XCTAssertTrue(nonFlat.isEmpty, "all scales should be 2-D; found non-flat: \(nonFlat.prefix(3))")

        // quantizedPaths surfaces the linears + embeddings for the quantize(filter:) step.
        let paths = ChatterboxLoader.quantizedPaths(w)
        XCTAssertGreaterThan(paths.count, 100)
        XCTAssertTrue(paths.contains(lin))
        XCTAssertTrue(paths.contains(emb))
    }

    func testCondsLoad() throws {
        let conds = try ChatterboxLoader.loadConds(files)
        XCTAssertEqual(conds.t3SpeakerEmb.shape, [1, 256])
        XCTAssertEqual(conds.t3CondPromptSpeechTokens.shape, [1, 375])
        XCTAssertEqual(conds.genEmbedding.shape, [1, 192])
        XCTAssertEqual(conds.genPromptFeat.shape, [1, 500, 80])
        XCTAssertEqual(conds.genPromptToken.shape, [1, 250])
    }
}
