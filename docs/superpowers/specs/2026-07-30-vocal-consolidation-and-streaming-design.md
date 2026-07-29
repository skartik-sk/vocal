# Vocal — Consolidation + Streaming Design

**Date:** 2026-07-30
**Status:** Draft (awaiting review)
**Baseline commit:** `623773f` (initial, on `master`, no co-author)

---

## 1. Goals

1. **Consolidate** the three currently-separate locations into one self-contained `vocal/` project (code in git; the 5.7 GB model stays external but reachable).
2. **Stream audio** so speech starts playing while generation continues, instead of waiting for a whole clause before any sound — dramatically cutting time-to-first-audio and shrinking inter-sentence gaps.
3. **Do not break** the currently-working right-click "Speak with Vocal" flow. Every change is reversible.

**Explicitly deferred (out of scope for now):** the MCP server, the JSON request/response protocol, and the persistent (load-once) worker mode. These are tracked as the "MCP phase" and will be built later, on top of the clean foundation this design produces.

---

## 2. Current state (why the pauses happen)

- **Three locations today:**
  - `vocal/` — Rust macOS Services host (`src/main.rs`). Grabs selected text, splits into sentences, spawns the sidecar, pipes sentences to its stdin, tees logs. **No audio code.**
  - `swift-qwen3-tts/` — the real Swift MLX engine; the sidecar binary is `.build/release/VocalWorker`.
  - `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/` — the 5.7 GB model (a HuggingFace git repo).
- **Audio never leaves the worker.** The worker reads sentences from stdin and plays 24 kHz mono Float32 PCM itself via `AVAudioEngine` + `AVAudioPlayerNode` (`VocalWorker/main.swift:16-70`). Rust only pipes text and reads logs.
- **Inter-clause pipelining already exists** (`queueAndPlay` is non-blocking `scheduleBuffer`), so clause N+1 *starts generating* while clause N plays.
- **Root cause of the gaps:** generation runs at **~0.6× realtime** (logs: "11.9s audio in 19.91s"). Each clause's generation outlasts the previous clause's playback, so the queue drains → silence. This is a **physics ceiling on this hardware**, not a missing queue.
- **Nothing is committed yet** before this work; baseline `623773f` captures the current source (build artifacts gitignored).

---

## 3. Workstream A — Consolidation (do first)

Pure restructure; the worker's current stdin-sentence protocol is **unchanged**.

### 3.1 Target layout
```
vocal/
├── Cargo.toml, Cargo.lock, src/main.rs   # Rust host (path config only changes)
├── Info.plist.ext
├── config.toml                            # NEW: single source of truth for paths/defaults
├── engine/                                # ← entire swift-qwen3-tts package moved in
│   ├── Package.swift, Package.resolved
│   ├── default.metallib                   # found via current_dir, moves with the package
│   └── Sources/{Qwen3TTS, VocalWorker, Qwen3TTSDemo}/, Tests/, docs/
├── model  →  /abs/.../Qwen3-TTS-…-8bit    # gitignored symlink (feels in-place, NOT copied)
└── (removed: vocal-engine/, main.swift, ai_worker.py, ai_worker_old.py)
```

### 3.2 Model handling
The 5.7 GB model **stays external** (HF git repo; cannot go in git). Two ways to reach it:
- A **gitignored symlink `vocal/model`** → the real path (so the project feels self-contained).
- **`VOCAL_MODEL_PATH`** env override (existing) wins if set; otherwise fall back to the symlink.

### 3.3 Config (removes duplicated hardcoded paths)
Today the model path is hardcoded in `src/main.rs:99` *and* `VocalWorker/main.swift:79`. Replace with **one** plain-text `vocal.config` file (simple `key = value` lines, hand-parsed in Rust — **no new crate**) holding: `model_path`, `engine_binary`, `speaker`, `language`, `instruct`, `temperature`. Rust is the single source of truth: it reads `vocal.config` and passes the values to the worker via the **existing env vars** (`VOCAL_MODEL_PATH`, etc.). The worker's own defaults remain only as a last-resort fallback for ad-hoc runs.

### 3.4 Dead code & build
- Delete `vocal-engine/`, `main.swift`, `ai_worker.py`, `ai_worker_old.py`.
- Document the build: `cd engine && swift build -c release`. Consider a tiny `build.sh` / Makefile.
- `.gitignore` (already updated) covers `/target/`, `.DS_Store`, `.build/`, `.swiftpm/`, `DerivedData/`, `*.wav`, and the `model` symlink.

### 3.5 Nothing-breaks for Workstream A
- Move the engine sources, rebuild `VocalWorker` at `engine/.build/release/VocalWorker`, update the Rust path config, and **smoke-test the right-click flow end-to-end before** deleting the old references and dead code.
- Since both ends (Rust + worker) live in one repo after the move, the wire format stays consistent. Same audio path. Only paths/config change.

---

## 4. Workstream B — Streaming (after A; Swift-internal)

All changes are inside the engine library + worker. **Rust is unchanged.** Built on code reading of `Qwen3.swift`, `Qwen3+Streaming.swift`, `SpeechTokenizer.swift`, `GenerationTypes.swift`.

### 4.1 What the code allows
- `generateCustomVoice` (`Qwen3.swift:783-962`): the Talker loop appends a complete 16-codebook step to `generatedCodes` **every iteration** (line 915). Mid-loop, `generatedCodes[0..<i]` is a valid partial code sequence.
- `decode` (`SpeechTokenizer.swift:823-836`) accepts **any** `seq_len` and computes valid length from non-padding first-codebook tokens (`validTokens * 1920`). **Decoding a partial prefix is legal.**
- Each codec token = `24000 / 1920` = **80 ms** of audio (12.5 Hz frame rate).
- Existing `generateStream` (`Qwen3+Streaming.swift`) yields `.token` events but only emits `.audio` once at the end — it is **token-streaming, not audio-streaming**. We add true audio streaming.

### 4.2 The obstacle
The decoder's `preTransformer` (`SpeechTokenizer.swift:763`) runs **with no attention mask** → bidirectional. Position *t*'s hidden state depends on future tokens. So the **right edge of any decoded window is unstable** until enough future tokens exist. (Convs are causal; only this transformer blocks single-token decode.)

### 4.3 Phase 0 — Profile (decisive, ~1 instrumented run)
Log, separately, for a fixed paragraph:
- steady-state **tokens/sec** (excluding prompt prefill),
- **decode time** for a full clause in isolation,
- the **prompt/prefill cost** per clause.

This tells us (a) the right chunk size K, and (b) whether the 0.6× is dominated by per-clause fixed overhead (→ Phase 2 worth it) or raw token cost (→ Phase 2 won't help). Decides the Phase 2 go/no-go with data, not guesswork.

### 4.4 Phase 1 — Chunked intra-clause streaming (the guaranteed win)
Add a new library method:

```swift
public func generateAudioStream(text:speaker:instruct:language:temperature:...)
    -> AsyncThrowingStream<AudioChunkEvent, Error>   // .chunk([Float]), .info(...), .end
```

Behavior:
- Runs the **same Talker loop** as `generateCustomVoice`, accumulating `generatedCodes`.
- Every **K ≈ 8-12 tokens (~0.6-1.0 s of audio)**, form a **sliding decode window** of the last ~**38 tokens + small lookahead** (the CloudWells/qwen3-tts-realtime-streaming recipe: 38-token context for voice stability).
- `decode()` the window, keep the **stable middle segment**, **linear-crossfade (~10 ms)** its head with the tail of the previously emitted audio, and yield the new samples.
- Track an "emitted samples" cursor to prevent gaps/overlaps.

Worker change (minimal):
```swift
// was: let audio = try await model.generate(...); player.queueAndPlay(audio.asArray(Float.self))
// now (generateAudioStream returns the stream synchronously, like generateStream):
let stream = model.generateAudioStream(text:..., speaker:..., instruct:..., language:..., temperature:...)
for try await event in stream {
    if case .chunk(let samples) = event { player.queueAndPlay(samples) }
}
```
The existing non-blocking `AVAudioEngine` queue + inter-clause pipelining are reused unchanged.

**Expected outcome:** first sound in **~1-3 s** (the warmup window) instead of ~19 s; much smaller gaps between clauses. Short bursts (≤ warmup) may finish generating before the window fills, so they gain little — the win is on medium/long utterances and continuous speech.

**Honest ceiling:** at 0.6× realtime, generation cannot keep the buffer full for long text; periodic pauses remain unless Phase 2 lifts the realtime factor above 1.0.

### 4.5 Phase 2 — Prompt-prefix KV caching (conditional on Phase 0)
Only pursued if Phase 0 shows per-clause fixed/prefill overhead is a large share of the 19.9 s. Today every clause re-runs `prepareGenerationInputs(...)` and `talker.makeCache()` (line 838) for the **identical** speaker/language/instruct prefix. Precompute that prefix's KV once, prime the cache per clause, and only process the per-clause text + generated codes. If the steady-state token rate is fast enough, this pushes the realtime factor toward/above 1.0 → near-gapless long reads. If Phase 0 shows raw token cost dominates, **skip Phase 2** and accept the 0.6× ceiling.

---

## 5. Safety, testing, rollback

- **Baseline:** `623773f` is the known-good rollback point.
- **One commit per phase** on top of the baseline; each is independently revertible.
- **Streaming never removes the proven path:** `model.generate(...)` stays; `generateAudioStream` is additive; the worker selects path via an env flag (`VOCAL_STREAM=1`). Misbehavior → flip back instantly.
- **Consolidation order:** move + rebuild + smoke-test right-click *before* deleting old paths/dead code.
- **Test harness:** a small CLI/script runs a fixed paragraph through both paths and reports TTFB, gaps, and tokens/sec; plus a manual listen test for boundary artifacts (clicks/stutter at chunk edges — the crossfade exists to suppress these).
- **Git hygiene:** build artifacts and the model stay out of git (`.gitignore`); no co-author trailer on commits (per user request).

---

## 6. Deferred to the MCP phase (later)

- JSON-lines request/response protocol (`speak`/`stop`/`quit` + `ready`/`started`/`chunk`/`finished`/`error`).
- Persistent (load-once) worker to avoid the 4.24 s reload per invocation.
- Concurrency/queueing for interleaved requests.
- The MCP server itself, so an agent can send short messages.

The consolidation here is structured so this phase is additive and low-risk later.

---

## 7. References
- Reference streaming impl (PyTorch): https://github.com/CloudWells/qwen3-tts-realtime-streaming — 3 s/38-token warmup, 0.6 s chunks, 38-token sliding window, 10 ms crossfade.
- Official Qwen3-TTS Dual-Track streaming guide: https://qwenlm-qwen3-tts.mintlify.app/guides/streaming
- Model card: https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
- Engine paper: `engine/docs/paper.pdf` (Qwen3-TTS architecture).
