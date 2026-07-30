import Foundation
@preconcurrency import MLX

extension Qwen3TTSModel {
    /// Streaming generation: yields `.chunk([Float])` roughly every `chunkTokens`
    /// (~0.6 s of audio) as codec tokens are produced, then a final `.info`.
    ///
    /// Each chunk is decoded from a bounded sliding window of codec codes with
    /// left-context (voice stability) and right-lookahead (so the chunk's right
    /// edge is stable despite the decoder's non-causal transformer), then glued
    /// to the previous chunk with a short linear crossfade. Playback can therefore
    /// start after the warmup window instead of after the whole utterance.
    ///
    /// Mirrors `generateCustomVoice`'s autoregressive loop; `generate(...)` is
    /// untouched and remains the whole-clip fallback.
    public func generateAudioStream(
        text: String,
        speaker: String,
        language: String = "auto",
        instruct: String? = nil,
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.05,
        maxTokens: Int = 2048,
        chunkTokens: Int = 8,          // ~0.64 s of audio per chunk
        contextTokens: Int = 38,       // left context for voice stability
        lookaheadTokens: Int = 8,      // right context so a chunk's right edge is stable
        overlapSeconds: Double = 0.01  // 10 ms linear crossfade between chunks
    ) -> AsyncThrowingStream<AudioStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Thread.detachNewThread {
                do {
                    let samplesPerToken = 1920           // 24000 Hz / 12.5 fps
                    let overlap = Int(overlapSeconds * 24000.0)

                    guard let talkerConfig = self.config.talkerConfig else {
                        throw Qwen3TTSError.modelNotInitialized("Talker config not available")
                    }
                    guard let speechTokenizer = self.speechTokenizer else {
                        throw Qwen3TTSError.modelNotInitialized("Speech tokenizer not loaded")
                    }
                    if let spkIdMap = talkerConfig.spkId,
                       !spkIdMap.keys.contains(speaker.lowercased()) {
                        throw Qwen3TTSError.invalidInput("Speaker '\(speaker)' not found.")
                    }

                    let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = self.prepareGenerationInputs(
                        text: text, language: language, speaker: speaker, instruct: instruct)
                    let targetTokenCount = self.tokenizer?.encode(text: text).count ?? text.count
                    let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))
                    let eosTokenId = talkerConfig.codecEosTokenId
                    let vocabSize = talkerConfig.vocabSize
                    var suppressTokens = [Int]()
                    for i in (vocabSize - 1024)..<vocabSize where i != eosTokenId {
                        suppressTokens.append(i)
                    }

                    let cache = self.talker.makeCache()
                    var generatedCodes: [[MLXArray]] = []
                    var generatedTokens: [Int] = []
                    var currentInput = inputEmbeds
                    var trailingIdx = 0

                    // Streaming state
                    var nextEmit = 0              // next token whose audio we will emit
                    var prevTail: [Float] = []    // tail of the last emit, for crossfade

                    // Decode codec codes [start..<end) → full mono [Float] window.
                    func decodeRange(_ start: Int, _ end: Int) -> [Float] {
                        var codesArray: [MLXArray] = []
                        for step in generatedCodes[start..<end] {
                            codesArray.append(MLX.concatenated(step, axis: 1))
                        }
                        let codes = MLX.stacked(codesArray, axis: 1)   // [1, len, 16]
                        let (wav, _) = speechTokenizer.decode(codes)
                        let trimmed = wav[0]
                        eval(trimmed)
                        return trimmed.asArray(Float.self)
                    }

                    // Emit audio for tokens [nextEmit ..< nextEmit+chunkTokens) using a
                    // window with left context + right lookahead, crossfaded to the prior tail.
                    func emitChunk() {
                        let chunkEnd = nextEmit + chunkTokens
                        let windowStart = max(0, nextEmit - contextTokens)
                        let windowEnd = min(generatedCodes.count, chunkEnd + lookaheadTokens)
                        guard windowEnd > windowStart else { return }
                        let wav = decodeRange(windowStart, windowEnd)
                        let emitEnd = min(chunkEnd, generatedCodes.count)
                        let r = StreamMath.samples(
                            forTokenRangeStart: nextEmit, end: emitEnd,
                            windowStart: windowStart, samplesPerToken: samplesPerToken)
                        let clamped = max(r.lowerBound, 0)..<min(r.upperBound, wav.count)
                        guard clamped.count > 0 else { return }
                        var segment = Array(wav[clamped])
                        if !prevTail.isEmpty {
                            segment = StreamMath.crossfade(prev: prevTail, next: segment, overlap: overlap)
                        }
                        continuation.yield(.chunk(segment))
                        prevTail = Array(segment.suffix(overlap))
                        nextEmit += chunkTokens
                    }

                    // ---- Autoregressive loop (same structure as generateCustomVoice) ----
                    var produced = 0
                    for _ in 0..<effectiveMaxTokens {
                        let (logits, hiddenStates) = self.talker(currentInput, cache: cache)
                        eval(logits, hiddenStates)

                        let nextToken = self.sampleToken(
                            logits, temperature: temperature, topK: topK, topP: topP,
                            repetitionPenalty: repetitionPenalty, generatedTokens: generatedTokens,
                            suppressTokens: suppressTokens, eosTokenId: eosTokenId)
                        let tokenValue = nextToken.item(Int.self)
                        generatedTokens.append(tokenValue)
                        if tokenValue == eosTokenId { break }

                        var codeTokens: [MLXArray] = [nextToken]
                        if let codePredictor = self.talker.codePredictor {
                            let seqLen = hiddenStates.dim(1)
                            let codeHidden = hiddenStates[0..., (seqLen - 1)..., 0...]
                            let codePredictorCache = codePredictor.makeCache()
                            for codeIdx in 0..<15 {
                                let codeInput: MLXArray
                                if codeIdx == 0 {
                                    let code0Embed = self.talker.getInputEmbeddings()(nextToken)
                                    codeInput = MLX.concatenated([codeHidden, code0Embed], axis: 1)
                                } else {
                                    let prevCode = codeTokens[codeIdx]
                                    codeInput = codePredictor.codecEmbedding[codeIdx - 1](prevCode)
                                }
                                let (codeLogits, _, _) = codePredictor(
                                    codeInput, cache: codePredictorCache, generationStep: codeIdx)
                                eval(codeLogits)
                                codeTokens.append(self.sampleToken(
                                    codeLogits, temperature: temperature, topK: topK, topP: topP))
                            }
                        }
                        generatedCodes.append(codeTokens)
                        produced += 1

                        let textEmbed: MLXArray
                        if trailingIdx < trailingTextHidden.dim(1) {
                            textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                            trailingIdx += 1
                        } else {
                            textEmbed = ttsPadEmbed
                        }
                        var codecEmbed = self.talker.getInputEmbeddings()(nextToken)
                        if let codePredictor = self.talker.codePredictor {
                            for (i, code) in codeTokens.dropFirst().enumerated() {
                                codecEmbed = codecEmbed + codePredictor.codecEmbedding[i](code)
                            }
                        }
                        currentInput = textEmbed + codecEmbed

                        // Emit every chunk once enough lookahead exists.
                        while generatedCodes.count >= nextEmit + chunkTokens + lookaheadTokens {
                            emitChunk()
                        }
                    }

                    // ---- Final tail: decode the whole thing, emit the not-yet-played remainder ----
                    if !generatedCodes.isEmpty {
                        var fullCodesArray: [MLXArray] = []
                        for step in generatedCodes {
                            fullCodesArray.append(MLX.concatenated(step, axis: 1))
                        }
                        let fullCodes = MLX.stacked(fullCodesArray, axis: 1)
                        let (fullWav, fullLens) = speechTokenizer.decode(fullCodes)
                        eval(fullWav)
                        let validLen = fullLens[0].item(Int.self)
                        let fullSamples: [Float] = fullWav[0].asArray(Float.self)
                        let tailLo = min(nextEmit * samplesPerToken, validLen)
                        if validLen > tailLo {
                            var tail = Array(fullSamples[tailLo..<validLen])
                            if !prevTail.isEmpty {
                                tail = StreamMath.crossfade(prev: prevTail, next: tail, overlap: overlap)
                            }
                            continuation.yield(.chunk(tail))
                        }
                    }

                    let info = AudioGenerationInfo(
                        promptTokenCount: self.tokenizer?.encode(text: text).count ?? 0,
                        generationTokenCount: produced,
                        prefillTime: 0, generateTime: 0, tokensPerSecond: 0,
                        peakMemoryUsage: Double(GPU.peakMemory) / 1e9)
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
