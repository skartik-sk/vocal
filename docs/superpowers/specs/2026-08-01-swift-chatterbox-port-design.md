# Native Swift Chatterbox-Turbo Port — Design

**Date:** 2026-08-01
**Branch:** `experiment/native-chatterbox` (cut from `experiment/chatterbox-turbo`)
**Status:** Approved (component-verified build, default-voice-only v1)

## Goal

A **pure-Swift/MLX** implementation of Chatterbox-Turbo (default voice) that drops into
the existing Vocal pipeline (`config.rs` → worker → MCP / right-click), with **zero Python
dependency** on this route. Target: cut RAM from ~710 MB (Python `mlx-audio` worker) to
~500–550 MB, while producing **identical audio** (same 4-bit weights, same math).

## Non-goals (v1)

- Voice cloning (needs VoiceEncoder + CAMPPlus + S3TokenizerV2; CAMPPlus weights are not
  even in the 4-bit checkpoint). Phase 2.
- Emotion / exaggeration / classifier-free-guidance (Turbo ignores these anyway).
- Chunked intra-clause streaming (whole-clip first; the existing streaming strategy ports later).
- Multilingual (Turbo is English-only via GPT-2 BPE).

## The model (ground truth from the cached 4-bit checkpoint)

Repo `mlx-community/chatterbox-turbo-4bit`. **397 MB**, single `model.safetensors`
(2750 tensors, 4-bit affine, group_size 64). Three components by weight prefix:

- `t3.*` (505) — T3, a **GPT-2-medium** decoder (24 layers, 1024 dim, 16 heads, learned
  `wpe` position embeddings, combined-QKV `c_attn`, gelu_new). Text → discrete speech tokens.
  Conditioned on a baked 256-d `speaker_emb` + 375-token `cond_prompt_speech_tokens`.
- `s3gen.*` (2230) — S3 generator: Conformer **encoder** (6+4 layers, relative-pos attention)
  → `encoder_proj` (512→80) → `ConditionalDecoder` UNet (1 down + 12 mid + 1 up block) that
  performs **mean-flow** CFM (2 Euler steps, linear schedule, no CFG) → **HiFTNet** vocoder
  (F0 predictor + neural harmonic source + Snake-activation resblocks + iSTFT → 24 kHz wav).
- `ve.*` (15) — Voice encoder (3 LSTMs + proj). **Unused for the default voice.**

**`conds.safetensors`** (161 KB) ships the entire default voice pre-computed:
`t3.speaker_emb [1,256]`, `t3.cond_prompt_speech_tokens [1,375]`, `gen.prompt_feat [1,500,80]`,
`gen.prompt_token [1,250]`, `gen.embedding [1,192]`, plus lengths and an ignored `emotion_adv`.
→ The default-voice port **loads these directly** and needs no voice encoder.

## Inference pipeline

```
text ─GPT2 BPE─▶ tokens ─T3 (GPT2, AR)─▶ speech tokens
                                            │ (conds: speaker_emb + 375-voice-prompt)
                                            ▼
        ─S3 Conformer encoder─▶ ─mean-flow CFM (2 steps)─▶ mel (80-bin, 50 Hz)
                                            │ (conds: prompt_feat + prompt_token + 192-d emb)
                                            ▼
                          ─HiFTNet vocoder─▶ 24 kHz wav
```

Constants: speech vocab 6563 (start 6561 / stop 6562), silence token 4299, `token_mel_ratio=2`,
mel hop 480 (50 Hz), vocoder upsample 8·5·3·4 = 480, output 24 kHz. T3 sampling defaults:
temperature 0.8, top_p 0.95, top_k 1000, repetition_penalty 1.2.

## Where it lives (full isolation — nothing breaks)

- New Swift targets inside the existing `engine/` SwiftPM package:
  - `Chatterbox` library — `engine/Sources/Chatterbox/` (the port).
  - `ChatterboxWorker` executable — `engine/Sources/ChatterboxWorker/main.swift`; **self-contained**
    (copies the `AudioPlayer` + stdin loop from `VocalWorker`; swaps only the `generate` call).
- `config.rs`: new `native_chatterbox` backend value → routes to
  `engine/.build/release/ChatterboxWorker` + passes the model dir via `CHATTERBOX_MODEL_PATH`.
- The Python `chatterbox`, Qwen3 `swift`, and `VocalWorker` paths are untouched. Default
  `backend` stays `chatterbox` (working) until M5 proves the native path out.

## Reuse from swift-qwen3-tts (the scaffold)

Lift near-verbatim: the `fromPretrained` weight-load recipe (`MLX.loadArrays` → detect `.scales`
→ `quantize(model:groupSize:bits:mode:filter:)` → `model.update(parameters:)`), `KVCache`,
LayerNorm, combined-QKV attention, `sampleToken` (rep-penalty/top-k/top-p), Conv1d/CausalConv
and ResBlock/Snake patterns (from `SpeechTokenizer.swift`), `melSpectrogram`, the `AudioPlayer`
+ stdin worker loop, and the `swift-transformers` GPT-2 BPE tokenizer (already a dependency).

Rewrite (the actual port): T3 conditioning/sampling orchestration, S3 Conformer encoder,
mean-flow `ConditionalDecoder`, HiFTNet vocoder, Chatterbox config structs.

## Verification strategy (component-by-component)

- Instrument the Python worker to dump reference intermediates for a fixed test sentence into a
  gitignored `engine/reference/`: `text_tokens`, `speech_tokens`, `mu`, `mel`, `wav`.
- Each Swift component loads the same weights, runs the same input, and is checked for numerical
  closeness **before** chaining.
- **Determinism:** T3 sampling is stochastic, so verification uses **greedy/argmax** decoding in
  both Python and Swift (runtime still samples normally). Speech tokens must then match exactly;
  mel/wav within a tolerance band.

## Build milestones (each independently testable + committable)

- **M0** — Branch + `Package.swift` + `ChatterboxWorker` skeleton (stdin loop + `AudioPlayer`,
  stub `generate` returns a tone) wired into `config.rs`/MCP/right-click. *Proves native plumbing.*
- **M1** — Weight + config + conds loading (build module tree, quantize, `update`; load conds).
  *Verify: loads clean, param counts sane. ✅ DONE — confirmed convs are float16 (unquantized);
  only linears/embeddings are 4-bit, handled by MLXNN's standard `quantize()`.*
- **M2** — T3 (GPT-2) + tokenizer. *Verify: speech tokens match Python (greedy).*
- **M3** — S3 encoder + mean-flow CFM decoder. *Verify: mel matches Python.*
- **M4** — HiFTNet vocoder. *Verify: wav matches Python + audible.*
- **M5** — Chain end-to-end + wire `native_chatterbox` backend. *Listen test + RAM measurement
  (payoff check vs 710 MB).*

## Risk register

- **HiFTNet vocoder** (F0 + neural source + iSTFT + Snake) — most architecturally unusual,
  highest transcription-error risk. Mitigation: verify mel→wav with a fixed mel input first.
- **4-bit Conv1d** — ~~conv weights ship quantized~~ **RESOLVED at M1 (non-issue):** verified from the
  checkpoint that convs are stored as plain **float16** with no scales. Only linears + embeddings
  are 4-bit (U32 weight + 2-D F16 scales/biases), consumed natively by MLXNN's `quantize()`. No
  dequantization is needed at all — better for RAM than assumed.
- **GPT-2 combined-QKV + learned pos-emb** — standard, low risk.

## Honest payoff

~150–250 MB RAM saved (native ~500–550 MB vs Python 710 MB). **Sound is identical** to the
Python worker — the robotic voice is a property of this model's default voice, not of Python vs
Swift. Only voice cloning (phase 2) or a different model changes the sound.
