<p align="center">
  <img src="src-tauri/icons/icon.png" width="120" alt="Vocal icon" />
</p>

<h1 align="center">Vocal</h1>

<p align="center">
  <b>On-device text-to-speech for macOS.</b><br/>
  Select any text → right-click → <i>Speak with Vocal</i>. No cloud, no internet — your words never leave your machine.
</p>

<p align="center">
  <a href="#demo">Demo</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#project-structure">Structure</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#backends">Backends</a>
</p>

---

Vocal turns Apple Silicon into a real-time, multilingual speech reader. A Rust
host lives in the macOS **Services** menu, drives a Swift **MLX** TTS engine on
the GPU, and exposes the whole thing to a **Tauri** manager GUI and an **MCP**
tool so AI agents can speak too.

- 🔒 **100% local** — inference runs on your Mac's Metal GPU. Nothing is sent to a server.
- 🌐 **Multilingual** — 12 languages (English, Hindi, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian + dialects) via Qwen3-TTS.
- 🖱️ **Three ways in** — macOS Service (right-click), the Vocal Manager desktop app, or an MCP `speak` tool for agents.
- ⚡ **Streaming + low memory** — the model loads once and is freed the instant it goes idle, so it sits quietly until you need it.
- 🎚️ **Pluggable backends** — Qwen3-TTS (Swift), Chatterbox (Python mlx-audio), or an experimental pure-Swift Chatterbox.

## Demo

Vocal ships **two on-device TTS engines**. Hear the same line rendered by each — all generated locally, no cloud.

**English** — *"Hello! I'm Vocal, an on-device text-to-speech engine that lives right inside your Mac's Services menu. No cloud, no internet — your words never leave your machine."*

**Chatterbox** — Vocal's flagship engine
<audio controls src="https://raw.githubusercontent.com/skartik-sk/vocal/main/assets/demo/chatterbox-english.wav"></audio>

**Qwen3-TTS** — multilingual (12 languages)
<audio controls src="https://raw.githubusercontent.com/skartik-sk/vocal/main/assets/demo/qwen-english.wav"></audio>

**हिन्दी (Hindi)** — *"नमस्ते! मैं वोकल हूँ, बिना इंटरनेट के आपके मैक पर बोलने वाला एक लोकल इंजन।"*

**Chatterbox** *(English-focused model, so Hindi is accented)*
<audio controls src="https://raw.githubusercontent.com/skartik-sk/vocal/main/assets/demo/chatterbox-hindi.wav"></audio>

**Qwen3-TTS**
<audio controls src="https://raw.githubusercontent.com/skartik-sk/vocal/main/assets/demo/qwen-hindi.wav"></audio>

## How it works

```
                       ┌─────────────────────────────────────┐
   right-click text ──▶│  Rust host  (Vocal.app / Service)   │
   Manager GUI ───────▶│  - reads vocal.config               │──┐
   MCP speak tool ────▶│  - builds worker launch spec        │  │  stdin (one sentence per line)
                       └─────────────────────────────────────┘  ▼
                                            ┌────────────────────────────────┐
                                            │  Swift MLX engine  (engine/)    │
                                            │  loads model once → AVAudioEngine│
                                            │  frees GPU when stdin closes    │
                                            └────────────────────────────────┘
```

1. You trigger speech from **any** of the three surfaces.
2. The Rust host reads `vocal.config`, picks the active backend, and spawns a worker process (`worker.rs` is the single source of truth for this).
3. Text is streamed to the worker over **stdin, one sentence per line**. The worker loads the model once, synthesises each sentence, and plays it through `AVAudioEngine`.
4. When the pipe closes, the worker exits and releases all GPU/RAM immediately — Vocal costs nothing while idle.

Logs always land in `/tmp/vocal.log` (even when macOS launches a terminal-less copy to handle a right-click).

## Project structure

```
vocal/
├── src/                  Rust host crate (`vocal`)
│   ├── main.rs           macOS Service app — NSServices “Speak with Vocal” handler
│   ├── worker.rs         backend-agnostic worker launch spec
│   ├── config.rs         vocal.config parser (root resolved at compile time)
│   ├── lib.rs
│   └── bin/vocal_mcp.rs  MCP server — exposes Vocal as an agent-callable `speak` tool
├── src-tauri/            Tauri v2 “Vocal Manager” desktop GUI
│   ├── src/              Rust commands the frontend calls
│   ├── capabilities/     Tauri permission surface
│   └── icons/            app icon (png / ico / icns)
├── frontend/             Manager UI (vanilla HTML / CSS / JS)
│   └── tabs: Manage · Test · Logs · MCP · Settings
├── engine/               Swift MLX TTS engine (SwiftPM package)
│   └── Sources/
│       ├── Qwen3TTS/     multilingual Qwen3-TTS (12 langs, 9 built-in voices)
│       ├── Chatterbox/   pure-Swift Chatterbox port
│       ├── VocalWorker/  stdin-driven sidecar (Qwen3-TTS) used by the host
│       └── ChatterboxWorker/ · ChatterboxMLWorker/
├── scripts/              chatterbox helpers (Python mlx-audio side)
├── assets/demo/          demo clips — Chatterbox + Qwen3-TTS, English + Hindi
├── Cargo.toml            host crate + cargo-bundle metadata (builds Vocal.app)
├── Info.plist.ext        registers the macOS Services menu item
└── vocal.config.example  copy to vocal.config and edit for your machine
```

## Setup

> **Requirements:** an Apple Silicon Mac (macOS 14+), Xcode command-line tools, Rust, Node for Tauri, and Swift 5.9+.

### 1. Download a model

Vocal defaults to the Qwen3-TTS **CustomVoice 4-bit** model (~808 MB) from the
`mlx-community` / `AtomGradient` HuggingFace repos. Pull one, e.g.:

```bash
huggingface-cli download AtomGradient/Qwen3-TTS-0.6B-CustomVoice-4bit-pruned-vocab-lite \
  --local-dir ~/models/Qwen3-TTS-CustomVoice-4bit
```

### 2. Configure

Copy the example config and point it at your model + engine directories:

```bash
cp vocal.config.example vocal.config
# then edit vocal.config: model_path, engine_dir, speaker, language
```

`vocal.config` is gitignored — it holds machine-specific absolute paths.

### 3. Build the Swift engine

```bash
cd engine
swift build -c release           # builds VocalWorker + ChatterboxWorker + …
# the engine needs default.metallib next to the package:
cp .build/release/default.metallib .   # or copy from /usr/lib
```

### 4. Build & install the Rust host (the Service)

```bash
cargo build --release
cargo bundle                      # produces Vocal.app with the Services item
# drag Vocal.app into /Applications, then log out/in (or run:
# /System/Library/CoreServices/pssDiagnose … ) so macOS picks up the Service
```

Now select text anywhere → right-click → **Services → Speak with Vocal**.

### 5. (Optional) Run the Vocal Manager GUI

```bash
cd src-tauri
cargo tauri dev                  # or: cargo tauri build
```

## Backends

Set `backend` in `vocal.config`:

| `backend`            | Engine                         | Needs Python? | Notes                                         |
|----------------------|--------------------------------|---------------|-----------------------------------------------|
| `swift` *(default)*  | Qwen3-TTS (Swift/MLX)          | no            | Multilingual, 9 voices, fast first-audio.     |
| `chatterbox`         | Chatterbox-Turbo (mlx-audio)   | yes           | Python `mlx-audio`; expressive, voice-cloning.|
| `native_chatterbox`  | Chatterbox (pure-Swift port)   | no            | Experimental work-in-progress port.           |

Built-in Qwen3-TTS voices: `Aiden`, `Ryan`, `Serena`, `Vivian`, `Sohee`,
`Ono_anna`, `Uncle_fu`, `Eric`, `Dylan`.

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — model by the Alibaba Qwen team
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Python implementation this engine was ported from
- [MLX](https://github.com/ml-explore/mlx) — Apple's on-device ML framework
- [Tauri](https://tauri.app) · [rmcp](https://github.com/modelcontextprotocol/rust-sdk)

## License

[MIT](LICENSE) © Singupalli Kartik
