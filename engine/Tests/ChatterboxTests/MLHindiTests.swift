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
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        XCTAssertNotNil(model.tokenizer)
        let toks = model.tokenizer!.tokenize(text: "नमस्ते, मैं हिंदी में बोल रहा हूँ।", languageID: "hi")
        print("[ML] hindi text tokens: \(toks)")
        // Python reference: [722, 1706, 1712, 1720, 1740, 1702, 1734, 7, 2, ...]
        XCTAssertEqual(toks.prefix(3), [722, 1706, 1712], "tokenizer should match Python MTL")
    }

    func testGeneratesHindiSpeechTokens() async throws {
        print("[ML] starting test")
        fflush(stdout)
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
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
        let lg = model.t3.speechHead(hidden[0..., 69, 0...])
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
        print("[ML] has EOS(6562): \(flat.contains(6562))")
        // Python reference: [6561, 3677, 6486, 1960, 3913, ...] — starts with BOS
        XCTAssertGreaterThan(flat.count, 10)
        XCTAssertEqual(Int(flat[0]), 6561)
        // Should not be degenerate (all zeros / one repeated token)
        let unique = Set(flat.map { Int($0) })
        XCTAssertGreaterThan(unique.count, 10, "output should be diverse speech tokens")
    }

    func testFlowProducesMel() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        // Use the exact tokens Python used (so the mel is directly comparable).
        let pyTokens: [Int32] = [6561, 3677, 6486, 1960, 3913, 6181, 4317, 659, 1946, 731,
                                 5401, 4269, 1761, 2222, 2388, 6258, 2360, 2519, 4632, 269,
                                 1480, 1833, 79, 916, 1882, 4595, 4314, 723, 5084, 4816,
                                 6039, 3789, 5302, 5194, 5346, 1226, 581, 4956, 3590, 2132,
                                 1805, 1806, 4299, 6405, 4218, 6486, 6486, 6405, 6405, 6405,
                                 6405, 6405, 6405, 6405, 6079, 6562]
        let tok = MLXArray(pyTokens).reshaped([1, -1])
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        let mel = model.flow.inference(token: tok, ref: ref, finalize: false)
        print("[ML] mel shape: \(mel.shape)")
        // Dump full mel for external comparison.
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_MEL_OUT"] {
            let flat = mel.asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
            print("[ML] dumped mel to \(outPath)")
        }
        let allMel = mel.asArray(Float.self)
        let mean = allMel.reduce(0) { $0 + $1 } / Float(allMel.count)
        let varSum = allMel.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
        let std = sqrt(varSum / Float(allMel.count))
        let mn = allMel.min() ?? 0, mx = allMel.max() ?? 0
        print("[ML] swift mel stats: mean=\(mean) std=\(std) min=\(mn) max=\(mx)")
        // Compare against Python reference if present.
        if let pyPath = ProcessInfo.processInfo.environment["PY_MEL_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            if pyArr.count == allMel.count {
                let pyMean = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let pyVar = pyArr.reduce(0) { $0 + ($1 - pyMean) * ($1 - pyMean) }
                let pyStd = sqrt(pyVar / Float(pyArr.count))
                var num: Float = 0
                for i in 0..<allMel.count { num += (allMel[i] - mean) * (pyArr[i] - pyMean) }
                let corr = num / (std * pyStd * Float(allMel.count))
                print("[ML] correlation vs Python: \(corr)")
                print("[ML] py mel stats: mean=\(pyMean) std=\(pyStd)")
            }
        }
        XCTAssertEqual(mel.shape, [1, 80, 106], "mel should match Python (1,80,106)")
        // Check the mel is not degenerate.
        let rms = sqrt(allMel.reduce(0) { $0 + $1 * $1 } / Float(allMel.count))
        XCTAssertTrue(rms > 0.5, "mel RMS should be meaningful, got \(rms)")
    }

    func testFlowEncoderMatchesPython() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let pyTokens: [Int32] = [6561, 3677, 6486, 1960, 3913, 6181, 4317, 659, 1946, 731,
                                 5401, 4269, 1761, 2222, 2388, 6258, 2360, 2519, 4632, 269,
                                 1480, 1833, 79, 916, 1882, 4595, 4314, 723, 5084, 4816,
                                 6039, 3789, 5302, 5194, 5346, 1226, 581, 4956, 3590, 2132,
                                 1805, 1806, 4299, 6405, 4218, 6486, 6486, 6405, 6405, 6405,
                                 6405, 6405, 6405, 6405, 6079, 6562]
        let token = MLXArray(pyTokens).reshaped([1, -1])
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        // Replicate flow.inference up to encoder_proj.
        let n = norm(ref.embedding, axes: [1], keepDims: true) + 1e-8
        _ = model.flow.spkEmbedAffine(ref.embedding / n)
        let fullToken = concatenated([ref.promptToken, token], axis: 1)
        let tokenLen = ref.promptTokenLen + MLXArray([token.dim(1)])
        let maxLen = Int(tokenLen.max().item(Int32.self))
        let seqRange = MLXArray((0..<maxLen).map { Int32($0) }).reshaped([1, maxLen])
        let mask = (seqRange .< tokenLen.expandedDimensions(axis: -1)).asType(.float32).expandedDimensions(axis: -1)
        let tokEmb = model.flow.inputEmbedding(clip(fullToken, min: 0, max: 6560)) * mask
        let (hRaw, _) = model.flow.encoder(tokEmb, xsLens: tokenLen)
        let h = model.flow.encoderProj(hRaw[0..., 0..<(hRaw.dim(1) - 6), 0...])
        print("[ML] encoder h shape: \(h.shape)")
        let head = h[0..., 0..., 0..<1].asArray(Float.self)
        print("[ML] h[0,:,0] head: \(Array(head.prefix(8)))")
        // Compare vs Python f32 dump.
        if let pyPath = ProcessInfo.processInfo.environment["PY_H_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = h.asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] encoder corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
        XCTAssertEqual(h.shape, [1, 420, 80])
    }

    func testDecoderFirstStepMatchesPython() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let pyTokens: [Int32] = [6561, 3677, 6486, 1960, 3913, 6181, 4317, 659, 1946, 731,
                                 5401, 4269, 1761, 2222, 2388, 6258, 2360, 2519, 4632, 269,
                                 1480, 1833, 79, 916, 1882, 4595, 4314, 723, 5084, 4816,
                                 6039, 3789, 5302, 5194, 5346, 1226, 581, 4956, 3590, 2132,
                                 1805, 1806, 4299, 6405, 4218, 6486, 6486, 6405, 6405, 6405,
                                 6405, 6405, 6405, 6405, 6079, 6562]
        let token = MLXArray(pyTokens).reshaped([1, -1])
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        // encoder up to h
        let n = norm(ref.embedding, axes: [1], keepDims: true) + 1e-8
        let spk = model.flow.spkEmbedAffine(ref.embedding / n)
        let fullToken = concatenated([ref.promptToken, token], axis: 1)
        let tokenLen = ref.promptTokenLen + MLXArray([token.dim(1)])
        let maxLen = Int(tokenLen.max().item(Int32.self))
        let seqRange = MLXArray((0..<maxLen).map { Int32($0) }).reshaped([1, maxLen])
        let mask = (seqRange .< tokenLen.expandedDimensions(axis: -1)).asType(.float32).expandedDimensions(axis: -1)
        let tokEmb = model.flow.inputEmbedding(clip(fullToken, min: 0, max: 6560)) * mask
        let (hRaw, _) = model.flow.encoder(tokEmb, xsLens: tokenLen)
        let h = model.flow.encoderProj(hRaw[0..., 0..<(hRaw.dim(1) - 6), 0...])   // (1,420,80)
        let melLen1 = ref.promptFeat.dim(1)
        let melLen2 = h.dim(1) - melLen1
        var conds = MLXArray.zeros([1, melLen1 + melLen2, 80], dtype: h.dtype)
        conds[0..., 0..<melLen1, 0...] = ref.promptFeat
        conds = conds.transposed(0, 2, 1)
        let totalLen = melLen1 + melLen2
        let cmask = MLXArray.ones([1, 1, totalLen], dtype: h.dtype)
        let mu = h.transposed(0, 2, 1)   // (1,80,420)
        let z = model.flow.decoder.randNoise[0..., 0..., 0..<mu.dim(2)]
        // Dump mu/cond/z for cross-checking vs Python.
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_MU_OUT"] {
            let flat = mu[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_COND_OUT"] {
            let flat = conds[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_Z0_OUT"] {
            let flat = z[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        let t = MLXArray([Float(1 - cos(0.0 * 0.5 * Double.pi))]).reshaped([1])
        let xIn = concatenated([z, z], axis: 0)
        let maskIn = concatenated([cmask, cmask], axis: 0)
        let muIn = concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)
        let tIn = concatenated([t, t], axis: 0)
        let spksIn = concatenated([spk, MLXArray.zeros(spk.shape)], axis: 0)
        let condIn = concatenated([conds, MLXArray.zeros(conds.shape)], axis: 0)
        // Pass raw x, mu, t, spks, cond (decoder builds the concat) — matches Python.
        let dphi = model.flow.decoder.estimator(x: xIn, mask: maskIn, mu: muIn,
                                                t: tIn, spks: spksIn, cond: condIn)
        // Manual chain for comparison (should be identical to dphi).
        let tEmbM = model.flow.decoder.estimator.timeMlp(sinusoidalPosEmb(tIn, 320))
        let spkM = broadcast(spksIn.expandedDimensions(axis: -1), to: [2, 80, 420])
        let xCatM = concatenated([xIn, muIn, spkM, condIn], axis: 1)
        var curM = model.flow.decoder.estimator.downBlocks[0].resnet(xCatM, mask: maskIn, timeEmb: tEmbM)
        var curTM = curM.transposed(0, 2, 1)
        for tb in model.flow.decoder.estimator.downBlocks[0].transformerBlocks { curTM = tb(curTM, mask: nil) }
        curM = curTM.transposed(0, 2, 1)
        if let dc = model.flow.decoder.estimator.downBlocks[0].downsample as? MLCausalConv1d { curM = dc(curM * maskIn) }
        for mb in model.flow.decoder.estimator.midBlocks {
            curM = mb.resnet(curM, mask: maskIn, timeEmb: tEmbM)
            var ctm = curM.transposed(0, 2, 1)
            for tb in mb.transformerBlocks { ctm = tb(ctm, mask: nil) }
            curM = ctm.transposed(0, 2, 1)
        }
        print("[ML] manual mid-chain shape: \(curM.shape)")
        print("[ML] dphi shape: \(dphi.shape)")
        // Manual up block + final, using down0 output as skip (recompute down0).
        var downM = model.flow.decoder.estimator.downBlocks[0].resnet(xCatM, mask: maskIn, timeEmb: tEmbM)
        var downTM = downM.transposed(0, 2, 1)
        for tb in model.flow.decoder.estimator.downBlocks[0].transformerBlocks { downTM = tb(downTM, mask: nil) }
        downM = downTM.transposed(0, 2, 1)
        if let dc = model.flow.decoder.estimator.downBlocks[0].downsample as? MLCausalConv1d { downM = dc(downM * maskIn) }
        // up: concat mid[..420] + skip(downM 420) -> 512ch
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_MID11_OUT"] {
            let flat = curM[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_DOWN0_OUT"] {
            let flat = downM[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        let upIn2 = concatenated([curM[0..., 0..., 0..<420], downM], axis: 1)
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_UPIN_OUT"] {
            let flat = upIn2[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        // Compare up resnet block1 and full resnet vs Python.
        let upB1 = model.flow.decoder.estimator.upBlocks[0].resnet.block1(upIn2, mask: maskIn)
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_UPB1_OUT"] {
            let flat = upB1[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
        }
        print("[ML] up block1 head: \(Array(upB1[0..., 0..., 0..<1].asArray(Float.self).prefix(4)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_UPB1_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = upB1[0].asArray(Float.self)
            var num: Float = 0, ps: Float = 0, ss: Float = 0
            let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
            let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
            for i in 0..<swArr.count {
                num += (swArr[i] - sm) * (pyArr[i] - pm)
                ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                ss += (swArr[i] - sm) * (swArr[i] - sm)
            }
            print("[ML] up block1 corr vs Python: \(num / sqrt(ps * ss))")
        }
        var upM = model.flow.decoder.estimator.upBlocks[0].resnet(upIn2, mask: maskIn, timeEmb: tEmbM)
        var upTM = upM.transposed(0, 2, 1)
        let upBias = maskToBias(maskIn, T: upTM.dim(1), dtype: upTM.dtype)
        for tb in model.flow.decoder.estimator.upBlocks[0].transformerBlocks { upTM = tb(upTM, mask: upBias) }
        upM = upTM.transposed(0, 2, 1)
        if let uc = model.flow.decoder.estimator.upBlocks[0].upsample as? MLCausalConv1d { upM = uc(upM * maskIn) }
        if let outPath = ProcessInfo.processInfo.environment["SWIFT_UP0_OUT"] {
            let flat = upM[0].asArray(Float.self)
            var data = Data()
            for v in flat { var f = v; data.append(Data(bytes: &f, count: 4)) }
            try data.write(to: URL(fileURLWithPath: outPath))
            print("[ML] dumped up0 to \(outPath)")
        }
        print("[ML] manual up0 head: \(Array(upM[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_UP0_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = upM[0].asArray(Float.self)
            var num: Float = 0, ps: Float = 0, ss: Float = 0
            let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
            let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
            for i in 0..<swArr.count {
                num += (swArr[i] - sm) * (pyArr[i] - pm)
                ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                ss += (swArr[i] - sm) * (swArr[i] - sm)
            }
            print("[ML] manual up0 corr vs Python: \(num / sqrt(ps * ss))")
        }
        // final
        var fin = model.flow.decoder.estimator.finalBlock(upM, mask: maskIn)
        fin = model.flow.decoder.estimator.finalProj(fin * maskIn)
        // Compare manual full vs dphi
        let fullM = fin.asArray(Float.self)
        let dArr = dphi.asArray(Float.self)
        var num: Float = 0, ps: Float = 0, ss: Float = 0
        let pm = fullM.reduce(0) { $0 + $1 } / Float(fullM.count)
        let sm = dArr.reduce(0) { $0 + $1 } / Float(dArr.count)
        for i in 0..<dArr.count {
            num += (dArr[i] - sm) * (fullM[i] - pm)
            ps += (fullM[i] - pm) * (fullM[i] - pm)
            ss += (dArr[i] - sm) * (dArr[i] - sm)
        }
        print("[ML] dphi vs manual-full corr: \(num / sqrt(ps * ss))")
        print("[ML] manual-full[0,:,0] head: \(Array(fin[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        let head = dphi[0..., 0..., 0..<1].asArray(Float.self)
        print("[ML] dphi[0,:,0] head: \(Array(head.prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_DPHI_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = dphi[0..<1].asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] dphi corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
        XCTAssertEqual(dphi.shape, [2, 80, 420])
    }

    func testDumpDecoderWeights() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let db = model.flow.decoder.estimator.downBlocks[0]
        let w = db.resnet.block1.conv.conv.weight
        print("[ML] resnet conv weight: \(w.shape) dtype=\(w.dtype)")
        let flat = w.asArray(Float.self)
        print("[ML] resnet first 8: \(Array(flat.prefix(8)))")
        print("[ML] block1 conv bias: \(Array(db.resnet.block1.conv.conv.bias!.asArray(Float.self).prefix(8)))")
        print("[ML] block1 norm w: \(Array(db.resnet.block1.norm.weight!.asArray(Float.self).prefix(8)))")
        print("[ML] block1 norm b: \(Array(db.resnet.block1.norm.bias!.asArray(Float.self).prefix(8)))")
        print("[ML] mlp_linear w: \(Array(db.resnet.mlpLinear.weight.asArray(Float.self).prefix(8)))")
        print("[ML] mlp_linear type: \(type(of: db.resnet.mlpLinear)) shape=\(db.resnet.mlpLinear.weight.shape)")
        print("[ML] res_conv w: \(Array(db.resnet.resConv.weight.asArray(Float.self).prefix(8)))")
        let qp = db.transformerBlocks[0].attn.queryProj.weight
        print("[ML] query_proj: \(qp.shape) dtype=\(qp.dtype)")
        // Mid block 0 weights
        let mb = model.flow.decoder.estimator.midBlocks[0]
        print("[ML] mid resnet block1 conv w: \(Array(mb.resnet.block1.conv.conv.weight.asArray(Float.self).prefix(4)))")
        print("[ML] mid block1 bias: \(Array(mb.resnet.block1.conv.conv.bias!.asArray(Float.self).prefix(4)))")
        print("[ML] mid block2 conv w: \(Array(mb.resnet.block2.conv.conv.weight.asArray(Float.self).prefix(4)))")
        print("[ML] mid res_conv w: \(Array(mb.resnet.resConv.weight.asArray(Float.self).prefix(4)))")
        print("[ML] mid norm1 w: \(Array(mb.resnet.block1.norm.weight!.asArray(Float.self).prefix(4)))")
        print("[ML] mid mlp type: \(type(of: mb.resnet.mlpLinear)) shape=\(mb.resnet.mlpLinear.weight.shape)")
        print("[ML] mid mlp_linear scales: \((mb.resnet.mlpLinear as? QuantizedLinear)?.scales.shape ?? [])")
        // Up block 0
        let ub = model.flow.decoder.estimator.upBlocks[0]
        print("[ML] final_proj weight: \(Array(model.flow.decoder.estimator.finalProj.conv.weight.asArray(Float.self).prefix(4)))")
        let f0p = model.mel2wav.f0Predictor
        print("[ML] f0 condnet count: \(f0p.condnet.count)")
        print("[ML] f0 condnet0 w head: \(Array(f0p.condnet[0].conv.weight.asArray(Float.self).prefix(8)))")
        print("[ML] (py: -0.03189264 -0.09846696 -0.02305185 0.05946546 0.09903435 -0.0128215)")
        print("[ML] f0 condnet0 b head: \(Array(f0p.condnet[0].conv.bias!.asArray(Float.self).prefix(8)))")
        print("[ML] (py: 0.08287138 0.02207253 0.106259 -0.06652258 0.04141199 -0.01206417)")
        print("[ML] f0 classifier type: \(type(of: f0p.classifier)) shape=\(f0p.classifier.weight.shape)")
        if let ql = f0p.classifier as? QuantizedLinear {
            let dq = MLX.dequantized(ql.weight, scales: ql.scales, biases: ql.biases,
                                     groupSize: ql.groupSize, bits: ql.bits, mode: .affine)
            print("[ML] f0 classifier dq head: \(Array(dq[0].asArray(Float.self).prefix(4)))")
        }
        print("[ML] final_block conv w: \(Array(model.flow.decoder.estimator.finalBlock.conv.conv.weight.asArray(Float.self).prefix(4)))")
        print("[ML] up resnet block1 conv shape: \(ub.resnet.block1.conv.conv.weight.shape)")
        print("[ML] up resnet block1 conv w: \(Array(ub.resnet.block1.conv.conv.weight.asArray(Float.self).prefix(4)))")
        print("[ML] up res_conv shape: \(ub.resnet.resConv.weight.shape)")
        print("[ML] up res_conv w: \(Array(ub.resnet.resConv.weight.asArray(Float.self).prefix(4)))")
        // All mid block conv weights
        for i in 0..<12 {
            let mbk = model.flow.decoder.estimator.midBlocks[i]
            let b1 = mbk.resnet.block1.conv.conv.weight[0, 0, 0].item(Float.self)
            let b2 = mbk.resnet.block2.conv.conv.weight[0, 0, 0].item(Float.self)
            let rc = mbk.resnet.resConv.weight[0, 0, 0].item(Float.self)
            let bb = mbk.resnet.block1.conv.conv.bias![0].item(Float.self)
            let mlp0: Float
            if let ql = mbk.resnet.mlpLinear as? QuantizedLinear {
                let dq = MLX.dequantized(ql.weight, scales: ql.scales, biases: ql.biases,
                                         groupSize: ql.groupSize, bits: ql.bits, mode: .affine)
                mlp0 = dq[0, 0].item(Float.self)
            } else {
                mlp0 = mbk.resnet.mlpLinear.weight[0, 0].item(Float.self)
            }
            print("[ML] mid\(i): b1=\(String(format: "%.6f", b1)) b2=\(String(format: "%.6f", b2)) res=\(String(format: "%.6f", rc)) bb=\(String(format: "%.6f", bb)) mlp=\(String(format: "%.6f", mlp0))")
        }
        if let ql = db.transformerBlocks[0].attn.queryProj as? QuantizedLinear {
            let dq = MLX.dequantized(
                ql.weight, scales: ql.scales, biases: ql.biases,
                groupSize: ql.groupSize, bits: ql.bits, mode: .affine)
            print("[ML] qproj dq shape: \(dq.shape) head: \(Array(dq[0].asArray(Float.self).prefix(6)))")
        }
        print("[ML] ff.layers.0 type: \(type(of: db.transformerBlocks[0].ff.layers[0]))")
        // Python reference: down_blocks_0.resnet.block1.conv.conv.weight
    }

    func testDecoderResnetMatchesPython() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let pyTokens: [Int32] = [6561, 3677, 6486, 1960, 3913, 6181, 4317, 659, 1946, 731,
                                 5401, 4269, 1761, 2222, 2388, 6258, 2360, 2519, 4632, 269,
                                 1480, 1833, 79, 916, 1882, 4595, 4314, 723, 5084, 4816,
                                 6039, 3789, 5302, 5194, 5346, 1226, 581, 4956, 3590, 2132,
                                 1805, 1806, 4299, 6405, 4218, 6486, 6486, 6405, 6405, 6405,
                                 6405, 6405, 6405, 6405, 6079, 6562]
        let token = MLXArray(pyTokens).reshaped([1, -1])
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        // Rebuild real mu (encoder output) like Python.
        let n = norm(ref.embedding, axes: [1], keepDims: true) + 1e-8
        let spk = model.flow.spkEmbedAffine(ref.embedding / n)
        let fullToken = concatenated([ref.promptToken, token], axis: 1)
        let tokenLen = ref.promptTokenLen + MLXArray([token.dim(1)])
        let maxLen = Int(tokenLen.max().item(Int32.self))
        let seqRange = MLXArray((0..<maxLen).map { Int32($0) }).reshaped([1, maxLen])
        let mask = (seqRange .< tokenLen.expandedDimensions(axis: -1)).asType(.float32).expandedDimensions(axis: -1)
        let tokEmb = model.flow.inputEmbedding(clip(fullToken, min: 0, max: 6560)) * mask
        let (hRaw, _) = model.flow.encoder(tokEmb, xsLens: tokenLen)
        let h = model.flow.encoderProj(hRaw[0..., 0..<(hRaw.dim(1) - 6), 0...])
        let melLen1 = ref.promptFeat.dim(1)
        let melLen2 = h.dim(1) - melLen1
        var conds = MLXArray.zeros([1, melLen1 + melLen2, 80], dtype: h.dtype)
        conds[0..., 0..<melLen1, 0...] = ref.promptFeat
        conds = conds.transposed(0, 2, 1)
        let totalLen = melLen1 + melLen2
        let cmask = MLXArray.ones([1, 1, totalLen], dtype: h.dtype)
        let mu = h.transposed(0, 2, 1)   // (1,80,420)
        let t = MLXArray([Float(1 - cos(0.0 * 0.5 * Double.pi))]).reshaped([1])
        // CFG batch-2, x = rand_noise (deterministic, matches Python seed-0)
        let x = concatenated([model.flow.decoder.randNoise[0..., 0..., 0..<totalLen],
                              model.flow.decoder.randNoise[0..., 0..., 0..<totalLen]], axis: 0)
        let maskIn = concatenated([cmask, cmask], axis: 0)
        let muIn = concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)
        let tIn = concatenated([t, t], axis: 0)
        let spksIn = concatenated([spk, MLXArray.zeros(spk.shape)], axis: 0)
        let condIn = concatenated([conds, MLXArray.zeros(conds.shape)], axis: 0)
        let dec = model.flow.decoder.estimator
        let tEmb = dec.timeMlp(sinusoidalPosEmb(tIn, dec.inChannels))
        if let pyPath = ProcessInfo.processInfo.environment["PY_TEMB_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = tEmb[0..<1].asArray(Float.self)
            let swHead = Array(swArr.prefix(8))
            print("[ML] swift t_emb head: \(swHead)")
            print("[ML] py t_emb head: \(Array(pyArr.prefix(8)))")
        }
        let spksBroadcast = broadcast(spksIn.expandedDimensions(axis: -1), to: [2, 80, totalLen])
        let xCat = concatenated([x, muIn, spksBroadcast, condIn], axis: 1)
        let h0 = dec.downBlocks[0].resnet(xCat, mask: maskIn, timeEmb: tEmb)
        if let pyPath = ProcessInfo.processInfo.environment["PY_DOWNRES_RN"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = h0[0..<1].asArray(Float.self)
            var num: Float = 0, ps: Float = 0, ss: Float = 0
            let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
            let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
            for i in 0..<swArr.count {
                num += (swArr[i] - sm) * (pyArr[i] - pm)
                ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                ss += (swArr[i] - sm) * (swArr[i] - sm)
            }
            print("[ML] downres(rn) corr vs Python: \(num / sqrt(ps * ss))")
            let maxd = zip(swArr, pyArr).map { abs($0.0 - $0.1) }.max() ?? 1
            print("[ML] downres(rn) max abs diff: \(maxd)")
        }
        // Full down block: resnet -> transformer -> downsample
        var xT = h0.transposed(0, 2, 1)
        for tb in dec.downBlocks[0].transformerBlocks { xT = tb(xT, mask: nil) }
        var down0 = xT.transposed(0, 2, 1)
        if let ds = dec.downBlocks[0].downsample as? Downsample1D {
            down0 = ds(down0 * maskIn)
        } else if let dc = dec.downBlocks[0].downsample as? MLCausalConv1d {
            down0 = dc(down0 * maskIn)
        }
        // Mid block 0
        let tEmbMid = dec.timeMlp(sinusoidalPosEmb(tIn, dec.inChannels))
        let midAdd = dec.midBlocks[0].resnet.mlpLinear(mish(tEmbMid))
        print("[ML] mid time-add[0,:4]: \(Array(midAdd[0].asArray(Float.self).prefix(4)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_MIDADD_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = midAdd[0].asArray(Float.self)
            let maxd = zip(swArr, pyArr).map { abs($0.0 - $0.1) }.max() ?? 1
            print("[ML] mid time-add max diff vs Python: \(maxd)")
        }
        let midResnet0 = dec.midBlocks[0].resnet(down0, mask: maskIn, timeEmb: tEmb)
        print("[ML] midResnet0 head: \(Array(midResnet0[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_MIDRES_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = midResnet0[0..<1].asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] midResnet corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
        var mid0 = midResnet0
        var midT = mid0.transposed(0, 2, 1)
        for tb in dec.midBlocks[0].transformerBlocks { midT = tb(midT, mask: nil) }
        mid0 = midT.transposed(0, 2, 1)
        // Compare mid outputs at blocks 0/5/11 against Python.
        var midChain = midResnet0
        var mi = 0
        for mb in dec.midBlocks {
            var hm = mb.resnet(midChain, mask: maskIn, timeEmb: tEmb)
            var hmt = hm.transposed(0, 2, 1)
            for tb in mb.transformerBlocks { hmt = tb(hmt, mask: nil) }
            midChain = hmt.transposed(0, 2, 1)
            if mi == 5 || mi == 11 {
                let key = mi == 5 ? "PY_MID5_PATH" : "PY_MID11_PATH"
                if let pyPath = ProcessInfo.processInfo.environment[key],
                   FileManager.default.fileExists(atPath: pyPath) {
                    let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
                    let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    let swArr = midChain[0..<1].asArray(Float.self)
                    var num: Float = 0, ps: Float = 0, ss: Float = 0
                    let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                    let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                    for i in 0..<swArr.count {
                        num += (swArr[i] - sm) * (pyArr[i] - pm)
                        ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                        ss += (swArr[i] - sm) * (swArr[i] - sm)
                    }
                    print("[ML] mid\(mi) corr vs Python: \(num / sqrt(ps * ss))")
                    let swStats = "mean=\(sm) std=\(sqrt(ss / Float(swArr.count)))"
                    let pyStats = "mean=\(pm) std=\(sqrt(ps / Float(pyArr.count)))"
                    print("[ML] mid\(mi) swift \(swStats) py \(pyStats)")
                }
            }
            mi += 1
        }
        mid0 = midChain
        // FULL decoder: down + mid + up + final (verify whole path).
        let fullDec = model.flow.decoder.estimator(x: x, mask: maskIn, mu: muIn,
                                                   t: tIn, spks: spksIn, cond: condIn)
        print("[ML] full decoder head: \(Array(fullDec[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_DPHI2_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = fullDec[0..<1].asArray(Float.self)
            var num: Float = 0, ps: Float = 0, ss: Float = 0
            let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
            let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
            for i in 0..<swArr.count {
                num += (swArr[i] - sm) * (pyArr[i] - pm)
                ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                ss += (swArr[i] - sm) * (swArr[i] - sm)
            }
            print("[ML] full decoder corr vs Python: \(num / sqrt(ps * ss))")
        }
        // Up block 0 (single up block; concat mid[trunc] with skip=hiddens[0]=down0)
        let ub0 = dec.upBlocks[0]
        let upIn = concatenated([mid0[0..., 0..., 0..<down0.dim(2)], down0], axis: 1)
        var up0 = ub0.resnet(upIn, mask: maskIn, timeEmb: tEmb)
        var upT = up0.transposed(0, 2, 1)
        for tb in ub0.transformerBlocks { upT = tb(upT, mask: nil) }
        up0 = upT.transposed(0, 2, 1)
        if let uc = ub0.upsample as? MLCausalConv1d { up0 = uc(up0 * maskIn) }
        print("[ML] up0 head: \(Array(up0[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_UP0_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = up0[0..<1].asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] up0 corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
        print("[ML] mid0 shape: \(mid0.shape)")
        let midHead = mid0[0..., 0..., 0..<1].asArray(Float.self)
        print("[ML] mid0 head: \(Array(midHead.prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_MID0_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = mid0[0..<1].asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] mid0 corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
        print("[ML] down0 shape: \(down0.shape)")
        let downHead = down0[0..., 0..., 0..<1].asArray(Float.self)
        print("[ML] down0 head: \(Array(downHead.prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_DOWN0_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = down0[0..<1].asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] down0 corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
        print("[ML] resnet0 shape: \(h0.shape)")
        let head = h0[0..., 0..., 0..<1].asArray(Float.self)
        print("[ML] resnet0 head: \(Array(head.prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_RESNET_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = h0[0..<1].asArray(Float.self)
            if pyArr.count == swArr.count {
                var num: Float = 0, ps: Float = 0, ss: Float = 0
                let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                for i in 0..<swArr.count {
                    num += (swArr[i] - sm) * (pyArr[i] - pm)
                    ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                    ss += (swArr[i] - sm) * (swArr[i] - sm)
                }
                print("[ML] resnet corr vs Python: \(num / sqrt(ps * ss))")
            }
        }
    }

    func testRandNoiseMatchesPython() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let rn = model.flow.decoder.randNoise
        let head = rn[0..., 0..., 0..<4].asArray(Float.self)
        print("[ML] swift rand_noise head: \(Array(head))")
        // Python: mx.random.seed(0); normal((1,80,15000)) -> [0.301, -0.626, -0.961, 0.499]
        let pyHead: [Float] = [0.30120817, -0.62615937, -0.96135813, 0.49892026]
        let diff = zip(Array(head), pyHead).map { abs($0.0 - $0.1) }.max() ?? 1
        print("[ML] rand_noise max diff vs Python (head): \(diff)")
        // Check a later slice too — load full Python reference if available.
        if let pyPath = ProcessInfo.processInfo.environment["PY_RN_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = rn.asArray(Float.self)
            if pyArr.count == swArr.count {
                let maxd = zip(swArr, pyArr).map { abs($0.0 - $0.1) }.max() ?? 1
                print("[ML] rand_noise FULL max diff vs Python: \(maxd)")
            }
        }
    }

    func testStageCorrelations() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let pyTokens: [Int32] = [6561, 3677, 6486, 1960, 3913, 6181, 4317, 659, 1946, 731,
                                 5401, 4269, 1761, 2222, 2388, 6258, 2360, 2519, 4632, 269,
                                 1480, 1833, 79, 916, 1882, 4595, 4314, 723, 5084, 4816,
                                 6039, 3789, 5302, 5194, 5346, 1226, 581, 4956, 3590, 2132,
                                 1805, 1806, 4299, 6405, 4218, 6486, 6486, 6405, 6405, 6405,
                                 6405, 6405, 6405, 6405, 6079, 6562]
        let token = MLXArray(pyTokens).reshaped([1, -1])
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        let n = norm(ref.embedding, axes: [1], keepDims: true) + 1e-8
        let spk = model.flow.spkEmbedAffine(ref.embedding / n)
        let fullToken = concatenated([ref.promptToken, token], axis: 1)
        let tokenLen = ref.promptTokenLen + MLXArray([token.dim(1)])
        let maxLen = Int(tokenLen.max().item(Int32.self))
        let seqRange = MLXArray((0..<maxLen).map { Int32($0) }).reshaped([1, maxLen])
        let mask = (seqRange .< tokenLen.expandedDimensions(axis: -1)).asType(.float32).expandedDimensions(axis: -1)
        let tokEmb = model.flow.inputEmbedding(clip(fullToken, min: 0, max: 6560)) * mask
        let (hRaw, _) = model.flow.encoder(tokEmb, xsLens: tokenLen)
        let h = model.flow.encoderProj(hRaw[0..., 0..<(hRaw.dim(1) - 6), 0...])
        let melLen1 = ref.promptFeat.dim(1)
        let melLen2 = h.dim(1) - melLen1
        var conds = MLXArray.zeros([1, melLen1 + melLen2, 80], dtype: h.dtype)
        conds[0..., 0..<melLen1, 0...] = ref.promptFeat
        conds = conds.transposed(0, 2, 1)
        let totalLen = melLen1 + melLen2
        let cmask = MLXArray.ones([1, 1, totalLen], dtype: h.dtype)
        let dec = model.flow.decoder.estimator
        let mu = h.transposed(0, 2, 1)
        let t = MLXArray([Float(1 - cos(0.0 * 0.5 * Double.pi))]).reshaped([1])
        let x = concatenated([model.flow.decoder.randNoise[0..., 0..., 0..<totalLen],
                              model.flow.decoder.randNoise[0..., 0..., 0..<totalLen]], axis: 0)
        let maskIn = concatenated([cmask, cmask], axis: 0)
        let muIn = concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)
        let tIn = concatenated([t, t], axis: 0)
        let spksIn = concatenated([spk, MLXArray.zeros(spk.shape)], axis: 0)
        let condIn = concatenated([conds, MLXArray.zeros(conds.shape)], axis: 0)
        let tEmb = dec.timeMlp(sinusoidalPosEmb(tIn, dec.inChannels))
        let spksBroadcast = broadcast(spksIn.expandedDimensions(axis: -1), to: [2, 80, totalLen])
        let xCat = concatenated([x, muIn, spksBroadcast, condIn], axis: 1)
        var cur = dec.downBlocks[0].resnet(xCat, mask: maskIn, timeEmb: tEmb)
        var curT = cur.transposed(0, 2, 1)
        for tb in dec.downBlocks[0].transformerBlocks { curT = tb(curT, mask: nil) }
        cur = curT.transposed(0, 2, 1)
        if let dc = dec.downBlocks[0].downsample as? MLCausalConv1d { cur = dc(cur * maskIn) }
        print("[ML] stage down0: \(corrTo(cur, env: "PY_STAGE_DOWN0_PATH"))")
        for (mi, mb) in dec.midBlocks.enumerated() {
            cur = mb.resnet(cur, mask: maskIn, timeEmb: tEmb)
            var ct = cur.transposed(0, 2, 1)
            for tb in mb.transformerBlocks { ct = tb(ct, mask: nil) }
            cur = ct.transposed(0, 2, 1)
            print("[ML] stage mid\(mi): \(corrTo(cur, env: "PY_STAGE_MID\(mi)_PATH"))")
        }
        _ = melLen2
    }

    /// Correlation of batch-0 of `arr` against the Python f32 dump at $env.
    private func corrTo(_ arr: MLXArray, env: String) -> Float {
        guard let pyPath = ProcessInfo.processInfo.environment[env],
              FileManager.default.fileExists(atPath: pyPath) else { return -2 }
        guard let pyData = try? Data(contentsOf: URL(fileURLWithPath: pyPath)) else { return -3 }
        let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let swArr = arr[0..<1].asArray(Float.self)
        guard pyArr.count == swArr.count else { return -4 }
        var num: Float = 0, ps: Float = 0, ss: Float = 0
        let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
        let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
        for i in 0..<swArr.count {
            num += (swArr[i] - sm) * (pyArr[i] - pm)
            ps += (pyArr[i] - pm) * (pyArr[i] - pm)
            ss += (swArr[i] - sm) * (swArr[i] - sm)
        }
        return num / sqrt(ps * ss)
    }

    func testEulerStep1MatchesPython() async throws {
        let model = try await ChatterboxML.fromPretrained("/tmp/chatterbox-4bit")
        let pyTokens: [Int32] = [6561, 3677, 6486, 1960, 3913, 6181, 4317, 659, 1946, 731,
                                 5401, 4269, 1761, 2222, 2388, 6258, 2360, 2519, 4632, 269,
                                 1480, 1833, 79, 916, 1882, 4595, 4314, 723, 5084, 4816,
                                 6039, 3789, 5302, 5194, 5346, 1226, 581, 4956, 3590, 2132,
                                 1805, 1806, 4299, 6405, 4218, 6486, 6486, 6405, 6405, 6405,
                                 6405, 6405, 6405, 6405, 6079, 6562]
        let token = MLXArray(pyTokens).reshaped([1, -1])
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        let n = norm(ref.embedding, axes: [1], keepDims: true) + 1e-8
        let spk = model.flow.spkEmbedAffine(ref.embedding / n)
        let fullToken = concatenated([ref.promptToken, token], axis: 1)
        let tokenLen = ref.promptTokenLen + MLXArray([token.dim(1)])
        let maxLen = Int(tokenLen.max().item(Int32.self))
        let seqRange = MLXArray((0..<maxLen).map { Int32($0) }).reshaped([1, maxLen])
        let mask = (seqRange .< tokenLen.expandedDimensions(axis: -1)).asType(.float32).expandedDimensions(axis: -1)
        let tokEmb = model.flow.inputEmbedding(clip(fullToken, min: 0, max: 6560)) * mask
        let (hRaw, _) = model.flow.encoder(tokEmb, xsLens: tokenLen)
        let h = model.flow.encoderProj(hRaw[0..., 0..<(hRaw.dim(1) - 6), 0...])
        let melLen1 = ref.promptFeat.dim(1)
        let melLen2 = h.dim(1) - melLen1
        var conds = MLXArray.zeros([1, melLen1 + melLen2, 80], dtype: h.dtype)
        conds[0..., 0..<melLen1, 0...] = ref.promptFeat
        conds = conds.transposed(0, 2, 1)
        let totalLen = melLen1 + melLen2
        let cmask = MLXArray.ones([1, 1, totalLen], dtype: h.dtype)
        let mu = h.transposed(0, 2, 1)
        let z = model.flow.decoder.randNoise[0..., 0..., 0..<totalLen]
        let tSpan = (0...10).map { Float($0) / 10 }.map { 1 - cos($0 * 0.5 * Float.pi) }
        let t = MLXArray([tSpan[0]]).reshaped([1])
        let dt = tSpan[1] - tSpan[0]
        let xIn = concatenated([z, z], axis: 0)
        let maskIn = concatenated([cmask, cmask], axis: 0)
        let muIn = concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)
        let tIn = concatenated([t, t], axis: 0)
        let spksIn = concatenated([spk, MLXArray.zeros(spk.shape)], axis: 0)
        let condIn = concatenated([conds, MLXArray.zeros(conds.shape)], axis: 0)
        let dphi = model.flow.decoder.estimator(x: xIn, mask: maskIn, mu: muIn,
                                                t: tIn, spks: spksIn, cond: condIn)
        let dCond = dphi[0..<1]
        let dUncond = dphi[1..<2]
        let d = 1.5 * dCond - 0.5 * dUncond
        let z1 = z + dt * d
        print("[ML] z1 head: \(Array(z1[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        // Full 10-step Euler with cfg rate 0.7 (Python inference_cfg_rate).
        var zk = z
        var tk = t
        var dtk = dt
        for step in 1...10 {
            let xk = concatenated([zk, zk], axis: 0)
            let mk = concatenated([cmask, cmask], axis: 0)
            let muk = concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)
            let tik = concatenated([tk, tk], axis: 0)
            let spkik = concatenated([spk, MLXArray.zeros(spk.shape)], axis: 0)
            let condik = concatenated([conds, MLXArray.zeros(conds.shape)], axis: 0)
            let dp = model.flow.decoder.estimator(x: xk, mask: mk, mu: muk, t: tik, spks: spkik, cond: condik)
            let dc = dp[0..<1]
            let du = dp[1..<2]
            let dd = 1.7 * dc - 0.7 * du
            zk = zk + dtk * dd
            tk = tk + dtk
            if step < 10 { dtk = tSpan[step + 1] - Float(tk.item(Float.self)) }
            if step == 2 || step == 5 || step == 10 {
                let key = "PY_Z\(step)_PATH"
                if let pyPath = ProcessInfo.processInfo.environment[key],
                   FileManager.default.fileExists(atPath: pyPath) {
                    let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
                    let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    let swArr = zk[0].asArray(Float.self)
                    var num: Float = 0, ps: Float = 0, ss: Float = 0
                    let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
                    let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
                    for i in 0..<swArr.count {
                        num += (swArr[i] - sm) * (pyArr[i] - pm)
                        ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                        ss += (swArr[i] - sm) * (swArr[i] - sm)
                    }
                    print("[ML] z\(step) corr vs Python: \(num / sqrt(ps * ss))")
                }
            }
        }
        print("[ML] z10 head: \(Array(zk[0..., 0..., 0..<1].asArray(Float.self).prefix(6)))")
        if let pyPath = ProcessInfo.processInfo.environment["PY_Z1_PATH"],
           FileManager.default.fileExists(atPath: pyPath) {
            let pyData = try Data(contentsOf: URL(fileURLWithPath: pyPath))
            let pyArr = pyData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let swArr = z1[0].asArray(Float.self)
            var num: Float = 0, ps: Float = 0, ss: Float = 0
            let pm = pyArr.reduce(0) { $0 + $1 } / Float(pyArr.count)
            let sm = swArr.reduce(0) { $0 + $1 } / Float(swArr.count)
            for i in 0..<swArr.count {
                num += (swArr[i] - sm) * (pyArr[i] - pm)
                ps += (pyArr[i] - pm) * (pyArr[i] - pm)
                ss += (swArr[i] - sm) * (swArr[i] - sm)
            }
            print("[ML] z1 corr vs Python: \(num / sqrt(ps * ss))")
            let maxd = zip(swArr, pyArr).map { abs($0.0 - $0.1) }.max() ?? 1
            print("[ML] z1 max abs diff: \(maxd)")
        }
        _ = melLen2
    }

    func testFullGenerateToWav() async throws {
        let model = try await ChatterboxML.fromPretrained(Self.modelPath)
        let toks = model.speechTokens(text: "नमस्ते, मैं हिंदी में बोल रहा हूँ।", language: "hi")
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        let mel = model.flow.inference(token: toks, ref: ref, finalize: false)
        let wav = model.mel2wav.generate(mel)
        print("[ML] wav samples: \(wav.count)")
        XCTAssertGreaterThan(wav.count, 10000, "should produce >10000 samples (~0.4s)")
        let peak = wav.map { abs($0) }.max() ?? 0
        let rms = sqrt(wav.reduce(0) { $0 + $1 * $1 } / Float(max(wav.count, 1)))
        print("[ML] wav peak=\(peak) rms=\(rms)")
        XCTAssertTrue(peak > 0.05, "audio should not be silent")
        // Write it out for listening.
        let dir = URL(fileURLWithPath: "/tmp/swift_hindi_out")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let wavURL = dir.appendingPathComponent("hindi_swift.wav")
        var pcm = Data()
        for s in wav {
            var v = Int16(max(-1, min(1, s)) * 32767)
            pcm.append(Data(bytes: &v, count: 2))
        }
        func writeWavHeader(_ data: Data, sampleRate: Int) -> Data {
            var out = Data()
            func put(_ s: String) { out.append(s.data(using: .ascii)!) }
            func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(Data($0)) } }
            func put16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { out.append(Data($0)) } }
            put("RIFF"); put32(UInt32(36 + data.count)); put("WAVE")
            put("fmt "); put32(16); put16(1); put16(1); put32(UInt32(sampleRate))
            put32(UInt32(sampleRate * 2)); put16(2); put16(16)
            put("data"); put32(UInt32(data.count)); out.append(data)
            return out
        }
        let wavData = writeWavHeader(pcm, sampleRate: 24000)
        try wavData.write(to: wavURL)
        print("[ML] wrote \(wavURL.path) (\(wavData.count) bytes)")
    }

    func testWavOnFixedMel() async throws {
        let model = try await ChatterboxML.fromPretrained(Self.modelPath)
        guard let p = ProcessInfo.processInfo.environment["SWIFT_MEL_PATH"],
              FileManager.default.fileExists(atPath: p) else { return }
        let data = try Data(contentsOf: URL(fileURLWithPath: p))
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let mel = MLXArray(floats).reshaped([1, 80, 106])
        let wav = model.mel2wav.generate(mel)
        let peak = wav.map { abs($0) }.max() ?? 0
        print("[ML] swift wav on fixed mel: count=\(wav.count) peak=\(peak)")
        print("[ML] (python decode peak was 0.57184327)")
        XCTAssertTrue(peak > 0.2, "should be loud")
    }

    func testAnnouncementAB() async throws {
        let model = try await ChatterboxML.fromPretrained(Self.modelPath)
        // Same text Python announced with: model name + Hindi sentence.
        let text = "यह पायथन चैटरबॉक्स बोल रहा है। नमस्ते, मैं हिंदी में बोल रहा हूँ।"
        let ids = model.tokenizer!.tokenize(text: text, languageID: "hi")
        let textTokens = MLXArray(ids.map { Int32($0) }).reshaped([1, -1])
        let cond = T3MLCond(speakerEmb: model.conds.t3SpeakerEmb,
                            emotionAdv: model.conds.t3EmotionAdv,
                            condPromptSpeechTokens: model.conds.t3CondPromptSpeechTokens)
        let toks = model.t3.inference(cond: cond, textTokens: textTokens,
                                      maxNewTokens: 120, temperature: 0.8)
        let ref = S3RefML(promptToken: model.conds.genPromptToken,
                          promptTokenLen: model.conds.genPromptTokenLen,
                          promptFeat: model.conds.genPromptFeat,
                          embedding: model.conds.genEmbedding)
        let mel = model.flow.inference(token: toks, ref: ref, finalize: false)
        let wav = model.mel2wav.generate(mel)
        let peak = wav.map { abs($0) }.max() ?? 0
        print("[ML] swift announce: samples=\(wav.count) peak=\(peak)")
        print("[ML] (python announce: 91200 samples, peak 0.845)")
        var pcm = Data()
        for s in wav {
            var v = Int16(max(-1, min(1, s)) * 32767)
            pcm.append(Data(bytes: &v, count: 2))
        }
        var out = Data()
        func put(_ s: String) { out.append(s.data(using: .ascii)!) }
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(Data($0)) } }
        func put16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { out.append(Data($0)) } }
        put("RIFF"); put32(UInt32(36 + pcm.count)); put("WAVE")
        put("fmt "); put32(16); put16(1); put16(1); put32(24000)
        put32(UInt32(24000 * 2)); put16(2); put16(16)
        put("data"); put32(UInt32(pcm.count)); out.append(pcm)
        try out.write(to: URL(fileURLWithPath: "/tmp/swift_announce.wav"))
        print("[ML] wrote /tmp/swift_announce.wav")
        XCTAssertTrue(peak > 0.05)
    }
}
