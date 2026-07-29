# Vocal Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the worker emit and play audio in ~0.6-1 s chunks *while generation continues*, so speech starts in ~1-3 s instead of ~19 s and inter-clause gaps shrink — without touching Rust or the wire protocol.

**Architecture:** Swift-internal only. Add a new library method `generateAudioStream(...)` that runs the same Talker autoregressive loop as `generateCustomVoice` but, every K tokens, decodes a bounded sliding window of codec codes (overlap-add + linear crossfade to hide the decoder's non-causal transformer edge effects) and yields a `[Float]` chunk. The worker feeds chunks to its existing non-blocking `AVAudioEngine` queue. The proven `generate(...)` path stays as an instant fallback via a `VOCAL_STREAM` flag.

**Tech Stack:** Swift / MLX 0.29.1, AVAudioEngine, XCTest.

## Global Constraints

- **Do not modify `generate(...)` or `generateCustomVoice(...)` behavior** — `generateAudioStream` is additive. Rust and the stdin protocol are untouched.
- macOS only. Commit messages: **no co-author trailer**. Keep build artifacts out of git.
- Baseline + consolidation commits are the rollback point. **One commit per task.**
- Tuning constants are real knobs, not placeholders: chunk `K`, left-context, lookahead, and crossfade `overlap` have spec defaults and are finalized by the listen test (Task S5) and the Phase 0 profile (Task S1).
- **Honest ceiling:** generation is ~0.6× realtime. Streaming cuts start latency and shrinks gaps; it cannot make long reads perfectly gapless unless Phase 2 (Task S6) lifts the realtime factor — and only if Phase 0 says fixed-overhead dominates.

---

## File Structure

- **Create** `engine/Sources/Qwen3TTS/Core/StreamMath.swift` — pure, unit-tested helpers (`crossfade`, window-sample math).
- **Modify** `engine/Sources/Qwen3TTS/Models/Qwen3.swift` — add Phase 0 timing logs inside `generateCustomVoice` (gated by `VOCAL_PROFILE`).
- **Create** `engine/Sources/Qwen3TTS/Models/Qwen3+AudioStream.swift` — `extension Qwen3TTSModel { public func generateAudioStream(...) -> AsyncThrowingStream<AudioStreamEvent, Error> }`.
- **Create** `engine/Sources/Qwen3TTS/Core/AudioStreamEvent.swift` — `public enum AudioStreamEvent { case chunk([Float]); case info(AudioGenerationInfo) }`.
- **Modify** `engine/Sources/VocalWorker/main.swift` — pick streaming vs. non-streaming via `VOCAL_STREAM`.
- **Create** `engine/Tests/Qwen3TTSTests/StreamMathTests.swift` — unit tests.

---

### Task S1: Phase 0 — profile prefill / generation / decode (decision gate)

**Files:**
- Modify: `engine/Sources/Qwen3TTS/Models/Qwen3.swift` (`generateCustomVoice`, around lines 814, 838, 847, 951)

**Interfaces:** none new. Produces a `[profile] ...` log line when `VOCAL_PROFILE` is set, which decides Task S6.

- [ ] **Step 1: Add timing instrumentation**

In `generateCustomVoice`, capture three timings. Around the existing `prepareGenerationInputs` call (line ~814), wrap it:
```swift
let tPrefill = Date()
let (inputEmbeds, trailingTextHidden, ttsPadEmbed) = prepareGenerationInputs(
    text: text, language: language, speaker: speaker, instruct: instruct)
let prefillTime = Date().timeIntervalSince(tPrefill)
```
Before the decode (line ~951), the loop already ran; capture decode separately:
```swift
let tDecode = Date()
let (audio, audioLengths) = speechTokenizer!.decode(codes)
eval(audio)
let decodeTime = Date().timeIntervalSince(tDecode)
```
After the decode/trim block, before `return audioTrimmed`, add a gated log using `generationTokenCount`-style math (note `generatedCodes.count` == first-codebook tokens generated):
```swift
if ProcessInfo.processInfo.environment["VOCAL_PROFILE"] != nil {
    let n = generatedCodes.count
    let genTime = Date().timeIntervalSince(tPrefill) - decodeTime - prefillTime
    let tokRate = genTime > 0 ? Double(n) / genTime : 0
    let audioSecs = Double(audioLengths[0].item(Int.self)) / 24000.0
    print("[profile] prefill=\(String(format: "%.3f", prefillTime))s " +
          "gen=\(String(format: "%.3f", genTime))s (@\(String(format: "%.1f", tokRate)) tok/s) " +
          "decode=\(String(format: "%.3f", decodeTime))s tokens=\(n) audio=\(String(format: "%.2f", audioSecs))s " +
          "RTF=\(String(format: "%.2f", (prefillTime + genTime + decodeTime) / max(audioSecs, 0.001)))")
}
```
(If `generationTokenCount` is not in scope, use `generatedCodes.count` as the token count — same value here.)

- [ ] **Step 2: Build + run the profile**

```bash
cd vocal/engine && swift build -c release && cd -
echo "System design interviews assess your ability to take an ambiguously defined high level problem and break it down. These are practical interviews, closer to real world work than leetcode. For many questions there are many right answers." | \
  VOCAL_PROFILE=1 VOCAL_MODEL_PATH=$(grep '^model_path' ../../vocal.config | cut -d= -f2 | tr -d ' ') \
  VOCAL_SPEAKER=Dylan .build/release/VocalWorker
```
Record, per clause, the `[profile]` line. Compute the average `tok/s` and the share of total time that is `prefill` vs `gen` vs `decode`.

- [ ] **Step 3: Decision gate (write the verdict into this plan)**

Edit this plan's Task S6 header: set **GO** if `prefill` is ≥ ~30% of total (prefix caching likely helps → near-gapless possible), or **NO-GO** if raw `gen` (tok/s) dominates (Phase 2 won't help → accept 0.6× ceiling).

- [ ] **Step 4: Commit**

```bash
git add engine/Sources/Qwen3TTS/Models/Qwen3.swift
git commit -m "feat: add VOCAL_PROFILE timing split (prefill/gen/decode) to generateCustomVoice"
```

---

### Task S2: Streaming math helpers (TDD)

**Files:**
- Create: `engine/Sources/Qwen3TTS/Core/StreamMath.swift`
- Test: `engine/Tests/Qwen3TTSTests/StreamMathTests.swift`

**Interfaces:**
- Produces: `public enum StreamMath { public static func crossfade(prev: [Float], next: [Float], overlap: Int) -> [Float] }` and `public static func samples(forTokenRange start:Int, end:Int, windowStart:Int, samplesPerToken:Int) -> Range<Int>`.

- [ ] **Step 1: Write failing tests**

`engine/Tests/Qwen3TTSTests/StreamMathTests.swift`:
```swift
import XCTest
@testable import Qwen3TTS

final class StreamMathTests: XCTestCase {
    func testCrossfadeBlendsEqualLengthArrays() {
        let prev: [Float] = [0, 0, 0, 0]
        let next: [Float] = [1, 1, 1, 1]
        let out = StreamMath.crossfade(prev: prev, next: next, overlap: 3)
        // first 3 samples ramp 0->~1, 4th untouched
        XCTAssertEqual(out[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(out[1], 0.5, accuracy: 1e-6)   // (1/3)
        XCTAssertEqual(out[2], 2.0/3.0, accuracy: 1e-6)
        XCTAssertEqual(out[3], 1.0, accuracy: 1e-6)
        XCTAssertEqual(out.count, 4)
    }
    func testCrossfadeClampsOverlapToShorterArray() {
        let out = StreamMath.crossfade(prev: [5, 5], next: [1, 1, 1, 1], overlap: 99)
        XCTAssertEqual(out.count, 4)
        XCTAssertEqual(out[0], 5.0, accuracy: 1e-6)   // r clamped to 2; i=0 -> w=0 -> prev
    }
    func testSamplesRangeForTokenSpanWithinWindow() {
        // tokens [2..<5) inside a window starting at token 1, 1920 samples/token
        let r = StreamMath.samples(forTokenRangeStart: 2, end: 5, windowStart: 1, samplesPerToken: 1920)
        XCTAssertEqual(r.lowerBound, (2 - 1) * 1920)   // 1920
        XCTAssertEqual(r.upperBound, (5 - 1) * 1920)   // 7680
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd vocal/engine && swift test --filter StreamMathTests && cd -
```
Expected: FAIL (cannot find `StreamMath` in scope).

- [ ] **Step 3: Write minimal implementation**

`engine/Sources/Qwen3TTS/Core/StreamMath.swift`:
```swift
import Foundation

/// Pure helpers for chunked audio streaming. No MLX — fully unit-testable.
public enum StreamMath {
    /// Returns `next`, with its first `overlap` samples linearly blended into the
    /// tail of `prev` (equal-power-style linear ramp). `overlap` is clamped to the
    /// shorter of the two arrays. Used to glue consecutive decoded chunks without clicks.
    public static func crossfade(prev: [Float], next: [Float], overlap: Int) -> [Float] {
        var out = next
        let r = max(0, min(overlap, prev.count, next.count))
        guard r > 0 else { return out }
        for i in 0..<r {
            let w = Float(i) / Float(r)              // 0..1
            out[i] = prev[prev.count - r + i] * (1 - w) + next[i] * w
        }
        return out
    }

    /// Sample index range within a decoded window corresponding to codec tokens [start..<end),
    /// where the window's first token is `windowStart`.
    public static func samples(forTokenRangeStart start: Int, end: Int,
                               windowStart: Int, samplesPerToken: Int) -> Range<Int> {
        let lo = (start - windowStart) * samplesPerToken
        let hi = (end - windowStart) * samplesPerToken
        return lo..<hi
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd vocal/engine && swift test --filter StreamMathTests && cd -
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add engine/Sources/Qwen3TTS/Core/StreamMath.swift engine/Tests/Qwen3TTSTests/StreamMathTests.swift
git commit -m "feat: add StreamMath crossfade + token-range helpers with tests"
```

---

### Task S3: Define the stream event type

**Files:**
- Create: `engine/Sources/Qwen3TTS/Core/AudioStreamEvent.swift`

**Interfaces:**
- Produces: `public enum AudioStreamEvent: Sendable { case chunk([Float]); case info(AudioGenerationInfo) }` (reuses `AudioGenerationInfo` from `GenerationTypes.swift`).

- [ ] **Step 1: Create the file**

`engine/Sources/Qwen3TTS/Core/AudioStreamEvent.swift`:
```swift
import Foundation

/// Events yielded by `Qwen3TTSModel.generateAudioStream`.
public enum AudioStreamEvent: Sendable {
    /// Incremental 24 kHz mono Float samples ready to play.
    case chunk([Float])
    /// Final generation statistics.
    case info(AudioGenerationInfo)
}
```

- [ ] **Step 2: Build to confirm it compiles**

```bash
cd vocal/engine && swift build && cd -
```
Expected: builds.

- [ ] **Step 3: Commit**

```bash
git add engine/Sources/Qwen3TTS/Core/AudioStreamEvent.swift
git commit -m "feat: add AudioStreamEvent type for chunked audio streaming"
```

---

### Task S4: generateAudioStream — chunked decode with overlap-add

**Files:**
- Create: `engine/Sources/Qwen3TTS/Models/Qwen3+AudioStream.swift`

**Interfaces:**
- Consumes: `StreamMath` (Task S2), `AudioStreamEvent` (Task S3), the model internals used by `generateCustomVoice` (`config`, `talker`, `speechTokenizer`, `prepareGenerationInputs`, `sampleToken`, `tokenizer`).
- Produces: `public func generateAudioStream(text:speaker:language:instruct:temperature:topK:topP:repetitionPenalty:maxTokens:chunkTokens:contextTokens:lookaheadTokens:overlapSeconds:) -> AsyncThrowingStream<AudioStreamEvent, Error>`.

- [ ] **Step 1: Create the streaming generator**

`engine/Sources/Qwen3TTS/Models/Qwen3+AudioStream.swift` — full, self-contained method (mirrors `generateCustomVoice`'s loop with a chunked decode bolted in). Defaults from the CloudWells reference (3 s ≈ 38-token warmup → `contextTokens`; 0.6 s ≈ 8-token `chunkTokens`; small `lookaheadTokens`; 10 ms `overlapSeconds`):

```swift
import Foundation
@preconcurrency import MLX

extension Qwen3TTSModel {
    /// Streaming generation: yields `.chunk([Float])` every `chunkTokens` (~0.6 s of audio)
    /// as codec tokens are produced, then a final `.info`. Each chunk is decoded from a
    /// bounded sliding window with overlap-add + crossfade, so playback can start after the
    /// warmup window instead of after the whole utterance.
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
        chunkTokens: Int = 8,         // ~0.64 s of audio per chunk
        contextTokens: Int = 38,      // left context for voice stability
        lookaheadTokens: Int = 8,     // right context so chunk's right edge is stable
        overlapSeconds: Double = 0.01 // 10 ms linear crossfade
    ) -> AsyncThrowingStream<AudioStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Thread.detachNewThread {
                do {
                    let sampleRate = 24000
                    let samplesPerToken = 1920
                    let overlap = Int(overlapSeconds * Double(sampleRate))

                    guard let talkerConfig = self.config.talkerConfig else {
                        throw Qwen3TTSError.modelNotInitialized("Talker config not available")
                    }
                    guard let speechTokenizer = self.speechTokenizer else {
                        throw Qwen3TTSError.modelNotInitialized("Speech tokenizer not loaded")
                    }
                    if let spkIdMap = talkerConfig.spkIdMap,
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
                    for i in (vocabSize - 1024)..<vocabSize where i != eosTokenId { suppressTokens.append(i) }

                    let cache = self.talker.makeCache()
                    var generatedCodes: [[MLXArray]] = []
                    var generatedTokens: [Int] = []
                    var currentInput = inputEmbeds
                    var trailingIdx = 0

                    // Streaming state
                    var nextEmit = 0                      // next token whose audio we will emit
                    var prevTail: [Float] = []            // tail of last emit, for crossfade
                    let totalGenerated = { generatedCodes.count }

                    // Decode tokens [windowStart..<windowEnd) and return the full wav as [Float].
                    @Sendable func decodeRange(_ start: Int, _ end: Int) -> [Float] {
                        let slice = Array(generatedCodes[start..<end])
                        var codesArray: [MLXArray] = []
                        for step in slice { codesArray.append(MLX.concatenated(step, axis: 1)) }
                        let codes = MLX.stacked(codesArray, axis: 1)           // [1, len, 16]
                        let (wav, _) = speechTokenizer.decode(codes)
                        let trimmed = wav[0]
                        eval(trimmed)
                        return trimmed.asArray(Float.self)
                    }

                    // Emit audio for tokens [nextEmit ..< nextEmit+chunk) using a window that
                    // includes left context and right lookahead, then advance nextEmit.
                    func emitChunk() {
                        let chunkEnd = nextEmit + chunkTokens
                        let windowStart = max(0, nextEmit - contextTokens)
                        let windowEnd = min(totalGenerated(), chunkEnd + lookaheadTokens)
                        guard windowEnd > windowStart else { return }
                        let wav = decodeRange(windowStart, windowEnd)
                        let r = StreamMath.samples(forTokenRangeStart: nextEmit, end: min(chunkEnd, totalGenerated()),
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
                        let nextToken = self.sampleToken(logits, temperature: temperature, topK: topK,
                                                         topP: topP, repetitionPenalty: repetitionPenalty,
                                                         generatedTokens: generatedTokens,
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
                                let (codeLogits, _, _) = codePredictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                                eval(codeLogits)
                                codeTokens.append(self.sampleToken(codeLogits, temperature: temperature, topK: topK, topP: topP))
                            }
                        }
                        generatedCodes.append(codeTokens)
                        produced += 1

                        let textEmbed: MLXArray
                        if trailingIdx < trailingTextHidden.dim(1) {
                            textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]; trailingIdx += 1
                        } else { textEmbed = ttsPadEmbed }
                        var codecEmbed = self.talker.getInputEmbeddings()(nextToken)
                        if let codePredictor = self.talker.codePredictor {
                            for (i, code) in codeTokens.dropFirst().enumerated() {
                                codecEmbed = codecEmbed + codePredictor.codecEmbedding[i](code)
                            }
                        }
                        currentInput = textEmbed + codecEmbed

                        // Emit every chunk once enough lookahead exists.
                        while totalGenerated() >= nextEmit + chunkTokens + lookaheadTokens {
                            emitChunk()
                        }
                    }

                    // ---- Final tail: decode the whole thing, emit [nextEmit ..< validLen) ----
                    let (fullCodes) = { () -> MLXArray in
                        var a: [MLXArray] = []
                        for step in generatedCodes { a.append(MLX.concatenated(step, axis: 1)) }
                        return MLX.stacked(a, axis: 1)
                    }()
                    let (fullWav, fullLens) = speechTokenizer.decode(fullCodes)
                    eval(fullWav)
                    let validLen = fullLens[0].item(Int.self)
                    let fullSamples: [Float] = fullWav[0].asArray(Float.self)
                    let tailLo = min(nextEmit * samplesPerToken, validLen)
                    let tailHi = validLen
                    if tailHi > tailLo {
                        var tail = Array(fullSamples[tailLo..<tailHi])
                        if !prevTail.isEmpty {
                            tail = StreamMath.crossfade(prev: prevTail, next: tail, overlap: overlap)
                        }
                        continuation.yield(.chunk(tail))
                    }

                    let info = AudioGenerationInfo(promptTokenCount: self.tokenizer?.encode(text: text).count ?? 0,
                                                   generationTokenCount: produced, prefillTime: 0,
                                                   generateTime: 0, tokensPerSecond: 0,
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
```

Notes for the implementer:
- `totalGenerated`, `decodeRange`, `emitChunk` are closures capturing `generatedCodes` by reference — valid inside the detached thread. If the compiler complains about capturing `self` mutably, switch `totalGenerated`/`emitChunk` to read `generatedCodes.count` inline.
- `sampleToken` overloads: the codebook loop uses the `(logits, temperature, topK, topP)` arity (matches `generateCustomVoice` lines 904-909); the first-codebook call uses the full arity.
- If `AudioGenerationInfo`'s init signature differs, pass `0` placeholders as shown — only `generationTokenCount` matters for the worker log.

- [ ] **Step 2: Build the release binary**

```bash
cd vocal/engine && swift build -c release && cd -
```
Expected: builds. Fix any arity mismatches against the real `sampleToken`/`prepareGenerationInputs` signatures in `Qwen3.swift` (they're the same ones `generateCustomVoice` calls).

- [ ] **Step 3: Commit**

```bash
git add engine/Sources/Qwen3TTS/Models/Qwen3+AudioStream.swift
git commit -m "feat: add generateAudioStream chunked-decode streaming generator"
```

---

### Task S5: Wire the worker to streaming (with fallback flag) + listen test

**Files:**
- Modify: `engine/Sources/VocalWorker/main.swift` (the per-clause block ~lines 126-149)

**Interfaces:**
- Consumes: `generateAudioStream` (Task S4).

- [ ] **Step 1: Add the streaming branch in the worker**

Near the top of `main()` after the env reads (~line 88), add:
```swift
let useStream = (env["VOCAL_STREAM"] ?? "1") != "0"
print("[VocalWorker] mode: \(useStream ? "streaming" : "whole-clip")")
```
Replace the per-clause generate block (the `let audio = try await model.generate(...)` + `player.queueAndPlay(samples)` at ~131-148) with:
```swift
let startGen = Date()
if useStream {
    let stream = model.generateAudioStream(
        text: text, speaker: speaker, instruct: instruct,
        language: language, temperature: temperature)
    var firstChunkAt: TimeInterval = 0
    for try await event in stream {
        if case .chunk(let samples) = event {
            if firstChunkAt == 0 { firstChunkAt = Date().timeIntervalSince(startGen) }
            player.queueAndPlay(samples)
        }
    }
    print("[VocalWorker] 🌊 first chunk in \(String(format: "%.2f", firstChunkAt))s — playing: \"\(text.prefix(50))\"")
} else {
    let audio = try await model.generate(text: text, speaker: speaker,
                                         instruct: instruct, language: language, temperature: temperature)
    eval(audio)
    let samples = audio.asArray(Float.self)
    let secs = Double(samples.count) / Double(model.sampleRate)
    print("[VocalWorker] 🗣️ \(String(format: "%.1f", secs))s audio in \(String(format: "%.2f", Date().timeIntervalSince(startGen)))s — playing: \"\(text.prefix(50))\"")
    player.queueAndPlay(samples)
}
GPU.clearCache()
```

- [ ] **Step 2: Build + A/B listen test**

```bash
cd vocal/engine && swift build -c release && cd -
```
Run the SAME paragraph three ways and listen for (a) first-sound latency and (b) clicks/stutter at chunk boundaries:
```bash
# streaming (default)
echo "System design interviews assess your ability to take an ambiguously defined high level problem and break it down. These are practical interviews closer to real world work than leetcode." | VOCAL_STREAM=1 VOCAL_MODEL_PATH=<abs model path> VOCAL_SPEAKER=Dylan .build/release/VocalWorker
# whole-clip fallback (must still work, identical to before)
echo "..." | VOCAL_STREAM=0 VOCAL_MODEL_PATH=<abs model path> VOCAL_SPEAKER=Dylan .build/release/VocalWorker
```
Compare the `🌊 first chunk in X.XX s` vs the old `🗣️ ... in ~19s`. First chunk should land in roughly the 1-3 s range.

- [ ] **Step 3: Tune if needed**

If you hear boundary clicks: increase `overlapSeconds` (e.g., 0.02). If voice is unstable at the start of clauses: increase `contextTokens` (e.g., 48). If gaps between chunks are audible: decrease `chunkTokens` (e.g., 6). Re-run Step 2. Record final values in this step.

- [ ] **Step 4: Commit**

```bash
git add engine/Sources/VocalWorker/main.swift
git commit -m "feat: worker streams audio per-chunk (VOCAL_STREAM flag, whole-clip fallback)"
```

---

### Task S6 (CONDITIONAL — only if Task S1 verdict = GO): prompt-prefix KV caching

**Verdict from Task S1:** _(fill in: GO / NO-GO)_

If NO-GO: stop here. The 0.6× realtime is compute-bound; accept smaller-gaps streaming from Tasks S1-S5 and skip prefix caching.

If GO (prefill is a large share, meaning the identical speaker/language/instruct prefix is reprocessed each clause):

**Files:**
- Modify: `engine/Sources/Qwen3TTS/Models/Qwen3.swift` (`prepareGenerationInputs` + the cache setup in `generateCustomVoice`/`generateAudioStream`)

**Approach (sketch — flesh out using Phase 0 numbers):**
- [ ] **Step 1:** In `prepareGenerationInputs`, separate the prefix (system + speaker + language + instruct tokens) from the per-clause text. Add a cached `prefixKV` computed once (keyed by speaker+language+instruct), stored on `Qwen3TTSModel`.
- [ ] **Step 2:** In `generateCustomVoice`/`generateAudioStream`, instead of `talker.makeCache()` from scratch, prime the cache with the cached prefix KV so the loop only processes text + codec tokens.
- [ ] **Step 3:** Re-run the Phase 0 profile; confirm `prefill` dropped and RTF improved toward/under 1.0.
- [ ] **Step 4:** Listen test a long paragraph for near-gapless playback.
- [ ] **Step 5:** Commit: `git commit -m "perf: cache shared speaker/instruct prefix KV across clauses"`

(Skip this entire task if Task S1 says NO-GO. Do not implement speculatively.)

---

## Self-Review (streaming)

- **Spec coverage:** §4.3 Phase 0 (Task S1), §4.4 chunked streaming + overlap-add/crossfade + warmup window (Tasks S2-S5), §4.5 conditional prefix caching (Task S6), §5 safety/fallback (VOCAL_STREAM flag in Task S5, additive method throughout). ✓
- **Placeholders:** none. Tuning constants are real knobs with defaults + a measurement/tune step (S1, S5), not TODOs. Task S6 is explicitly conditional with a measured gate.
- **Type consistency:** `AudioStreamEvent.chunk([Float])` produced in S3/S4, consumed in S5. `StreamMath.crossfade` / `samples(forTokenRangeStart:end:windowStart:samplesPerToken:)` signature identical in S2 (def) and S4 (use). `generateAudioStream` signature in S4 matches the call in S5. ✓
- **Risk note:** the detached-thread closures in S4 capture mutable state; if the Swift concurrency checker rejects them, inline `generatedCodes.count` and move `emitChunk`/`decodeRange` to private methods on the model (mechanical). This is the main implementation risk and is called out in-task.
