# Vocal Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the separate `swift-qwen3-tts` engine into `vocal/engine/`, keep the 5.7 GB model external but configurable, remove dead code, and make the right-click "Speak with Vocal" flow work from one self-contained repo — with zero behavior change.

**Architecture:** Pure restructure. The Rust host (`src/main.rs`) becomes the single source of truth for paths via a tiny gitignored `vocal.config` (absolute paths, because macOS launches Services with an unpredictable cwd). It passes model/voice settings to the worker over the **existing** env-var + stdin-sentence protocol, which is unchanged. Audio still plays inside the Swift worker.

**Tech Stack:** Rust 2024 (objc2, no new deps), Swift Package Manager (MLX 0.29.1), macOS AVAudioEngine / NSServices.

## Global Constraints

- macOS only (Metal, AVAudioEngine, NSServices). Commit messages have **no co-author trailer**.
- Out of git: `target/`, `.build/`, `.swiftpm/`, `DerivedData/`, `*.wav`, `.DS_Store`, **`vocal.config`** (machine-specific absolute paths). Commit `vocal.config.example` instead.
- The Rust↔worker wire format (stdin sentences + env vars) **must not change** — the JSON protocol / persistent worker are deferred to the MCP phase.
- Rust stays audio-free (no audio crates).
- Baseline `623773f` is the rollback point. **One commit per task.**
- Deviation from spec (noted): use absolute paths in `vocal.config` instead of a relative `model` symlink, because macOS Services cwd is unpredictable. Same outcome (model external, configurable, untracked).

---

## File Structure

- **Create** `src/config.rs` — tiny `key = value` parser + `VocalConfig` (model_path, engine_dir, speaker, language, instruct, temperature). Pure, unit-tested.
- **Modify** `src/main.rs` — `mod config;`, load `vocal.config` (fallback to defaults), replace the hardcoded literals at `main.rs:97-99` with config-derived values.
- **Create** `vocal.config` (gitignored) and `vocal.config.example` (committed).
- **Move** `../swift-qwen3-tts/{Package.swift,Package.resolved,default.metallib,README.md,Sources,Tests,docs}` → `vocal/engine/` (exclude `.build`, `.git`, `DerivedData`, `.swiftpm`).
- **Delete** `vocal-engine/`, `main.swift`, `ai_worker.py`, `ai_worker_old.py`.
- **Modify** `.gitignore` — add `/vocal.config`.

---

### Task 1: Config parser module (TDD)

**Files:**
- Create: `src/config.rs`
- Modify: `src/main.rs` (add `mod config;` near the top, after the `use` lines ~line 13)

**Interfaces:**
- Produces: `pub struct VocalConfig { pub model_path: String, pub engine_dir: String, pub speaker: String, pub language: String, pub instruct: Option<String>, pub temperature: String }`, `impl Default for VocalConfig`, `pub fn load(path: &Path) -> VocalConfig`, `pub fn engine_binary(&self) -> PathBuf`, `pub fn engine_cwd(&self) -> PathBuf`.

- [ ] **Step 1: Write the failing tests**

Append to `src/config.rs` (full file, including impls below):

```rust
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct VocalConfig {
    pub model_path: String,
    pub engine_dir: String,
    pub speaker: String,
    pub language: String,
    pub instruct: Option<String>,
    pub temperature: String,
}

impl Default for VocalConfig {
    fn default() -> Self {
        Self {
            model_path: "/Users/singupallikartik/Developer/fun-projects/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit".into(),
            engine_dir: "/Users/singupallikartik/Developer/fun-projects/vocal/engine".into(),
            speaker: "Dylan".into(),
            language: "English".into(),
            instruct: Some("be very Fast, Serious, and not skip any word like you are reading audiobook".into()),
            temperature: "0.8".into(),
        }
    }
}

impl VocalConfig {
    pub fn load(path: &Path) -> VocalConfig {
        let mut cfg = VocalConfig::default();
        if let Ok(text) = fs::read_to_string(path) {
            for (key, value) in parse_kv(&text) {
                match key.as_str() {
                    "model_path" => cfg.model_path = value,
                    "engine_dir" => cfg.engine_dir = value,
                    "speaker" => cfg.speaker = value,
                    "language" => cfg.language = value,
                    "instruct" => cfg.instruct = if value.is_empty() { None } else { Some(value) },
                    "temperature" => cfg.temperature = value,
                    _ => {} // ignore unknown keys (forward-compatible)
                }
            }
        }
        cfg
    }

    pub fn engine_binary(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir).join(".build/release/VocalWorker")
    }

    pub fn engine_cwd(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir)
    }
}

fn parse_kv(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_kv_handles_comments_blanks_and_unknown() {
        let text = "# a comment\nmodel_path = /x/y\n\nspeaker=Dylan\n  temperature  =  0.7  \nbogus = z\n";
        let map: std::collections::HashMap<String, String> =
            parse_kv(text).into_iter().collect();
        assert_eq!(map.get("model_path"), Some(&"/x/y".to_string()));
        assert_eq!(map.get("speaker"), Some(&"Dylan".to_string()));
        assert_eq!(map.get("temperature"), Some(&"0.7".to_string()));
        assert_eq!(map.get("bogus"), Some(&"z".to_string()));
    }

    #[test]
    fn load_overrides_defaults_and_handles_missing_file() {
        // Missing file -> all defaults.
        let none = VocalConfig::load(Path::new("/does/not/exist/vocal.config"));
        assert_eq!(none.speaker, "Dylan");

        // Temp file with overrides.
        let p = std::env::temp_dir().join("vocal_config_test.cfg");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "speaker = Aiden").unwrap();
            writeln!(f, "instruct =").unwrap(); // empty -> None
        }
        let cfg = VocalConfig::load(&p);
        assert_eq!(cfg.speaker, "Aiden");
        assert_eq!(cfg.instruct, None);
        // Unspecified fields keep defaults.
        assert_eq!(cfg.temperature, "0.8");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn engine_binary_and_cwd_derive_from_engine_dir() {
        let mut cfg = VocalConfig::default();
        cfg.engine_dir = "/tmp/engine".into();
        assert_eq!(cfg.engine_binary(), PathBuf::from("/tmp/engine/.build/release/VocalWorker"));
        assert_eq!(cfg.engine_cwd(), PathBuf::from("/tmp/engine"));
    }
}
```

- [ ] **Step 2: Wire the module into main.rs**

In `src/main.rs`, immediately after the `use` block (after line 13 `use std::thread;`), add:

```rust
mod config;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test config`
Expected: PASS (3 tests). (Implementation is already in the file above, so tests pass immediately — this is the minimal correct impl.)

- [ ] **Step 4: Commit**

```bash
git add src/config.rs src/main.rs
git commit -m "feat: add VocalConfig key=value parser with defaults"
```

---

### Task 2: Create vocal.config + wire main.rs to use it

**Files:**
- Create: `vocal.config.example` (committed), `vocal.config` (gitignored)
- Modify: `src/main.rs:97-103` (spawn block), `.gitignore`

**Interfaces:**
- Consumes: `config::VocalConfig` from Task 1.
- Produces: a `VocalConfig` loaded at the start of `handle_speak_text`'s spawned thread, used to build the `Command`.

- [ ] **Step 1: Create vocal.config.example (committed template)**

`vocal.config.example`:
```
# Vocal runtime config (copy to vocal.config and edit). vocal.config is gitignored.
# macOS launches the Services app with an unpredictable cwd, so use ABSOLUTE paths.

model_path = /Users/singupallikartik/Developer/fun-projects/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
engine_dir = /Users/singupallikartik/Developer/fun-projects/vocal/engine
speaker = Dylan
language = English
instruct = be very Fast, Serious, and not skip any word like you are reading audiobook
temperature = 0.8
```

- [ ] **Step 2: Create the real vocal.config (same content, gitignored)**

Copy `vocal.config.example` to `vocal.config`:
```bash
cp vocal.config.example vocal.config
```

- [ ] **Step 3: Gitignore vocal.config**

Append to `.gitignore`:
```
# Machine-specific runtime config (absolute paths)
/vocal.config
```

- [ ] **Step 4: Replace the hardcoded spawn block in main.rs**

In `src/main.rs`, inside the `thread::spawn(move || { ... })` block (currently the `let mut child = Command::new(...)` at lines ~97-103), replace the hardcoded `.env(...)`/`Command::new`/`current_dir` with config-derived values. At the top of the spawned closure (right after `use std::io::Write;` ~line 88), load the config:

```rust
let cfg = config::VocalConfig::load(std::path::Path::new("vocal.config"));
```

Then replace the `Command::new(...)` block with:

```rust
let mut child = Command::new(cfg.engine_binary())
    .current_dir(cfg.engine_cwd())
    .env("VOCAL_MODEL_PATH", &cfg.model_path)
    .env("VOCAL_SPEAKER", &cfg.speaker)
    .env("VOCAL_LANGUAGE", &cfg.language)
    .env("VOCAL_INSTRUCT", cfg.instruct.as_deref().unwrap_or(""))
    .env("VOCAL_TEMPERATURE", &cfg.temperature)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()
    .expect("Failed to start Native Swift Engine");
```

Note: `VOCAL_INSTRUCT` now sends `""` when absent. The worker reads `env["VOCAL_INSTRUCT"]` (a non-nil `Some("")`), so also adjust the worker in Task 4 to treat empty string as "no instruct". (Worker default currently is `env["VOCAL_INSTRUCT"]` with no fallback — an empty string would be passed through; see Task 4 Step for the one-line guard.)

- [ ] **Step 5: Verify it compiles**

Run: `cargo build`
Expected: builds cleanly.

- [ ] **Step 6: Commit**

```bash
git add vocal.config.example .gitignore src/main.rs
git commit -m "feat: drive worker spawn from vocal.config (absolute paths)"
```

---

### Task 3: Move the engine into vocal/engine/ and build it

**Files:**
- Move (copy then verify): `../swift-qwen3-tts/{Package.swift,Package.resolved,default.metallib,README.md,Sources,Tests,docs}` → `vocal/engine/`

**Interfaces:** none new. Produces `vocal/engine/.build/release/VocalWorker`.

- [ ] **Step 1: Copy engine sources (exclude build artifacts and the engine's own .git)**

From the repo root:
```bash
mkdir -p vocal/engine
cp -R ../swift-qwen3-tts/Package.swift ../swift-qwen3-tts/Package.resolved \
      ../swift-qwen3-tts/default.metallib ../swift-qwen3-tts/README.md vocal/engine/
cp -R ../swift-qwen3-tts/Sources vocal/engine/Sources
cp -R ../swift-qwen3-tts/Tests vocal/engine/Tests
cp -R ../swift-qwen3-tts/docs vocal/engine/docs
```
Verify no `.build`/`DerivedData`/`.git` came along:
```bash
ls -la vocal/engine            # should show NO .build, NO .git
find vocal/engine -name .build -o -name DerivedData | head   # should be empty
```

- [ ] **Step 2: Build the release binary in the new location**

```bash
cd vocal/engine && swift build -c release && cd -
```
Expected: succeeds; `vocal/engine/.build/release/VocalWorker` exists:
```bash
test -x vocal/engine/.build/release/VocalWorker && echo OK
```

- [ ] **Step 3: Confirm .gitignore already covers engine artifacts**

The existing `.gitignore` has `.build/`, `.swiftpm/`, `DerivedData/` (no leading slash → match at any depth), so `vocal/engine/.build` etc. are already ignored. Verify:
```bash
git status --short vocal/engine | grep -E '\.build|DerivedData' && echo "LEAK" || echo "clean"
```

- [ ] **Step 4: Commit the engine sources**

```bash
git add vocal/engine
git commit -m "feat: move swift-qwen3-tts engine into vocal/engine"
```

---

### Task 4: End-to-end smoke test, worker guard, dead-code removal

**Files:**
- Modify: `vocal/engine/Sources/VocalWorker/main.swift:84` (empty-instruct guard)
- Delete: `vocal-engine/`, `main.swift`, `ai_worker.py`, `ai_worker_old.py`

**Interfaces:** none new.

- [ ] **Step 1: Guard empty VOCAL_INSTRUCT in the worker**

In `vocal/engine/Sources/VocalWorker/main.swift` line ~84, change:
```swift
let instruct = env["VOCAL_INSTRUCT"]
```
to:
```swift
let rawInstruct = env["VOCAL_INSTRUCT"]
let instruct = (rawInstruct?.trimmingCharacters(in: .whitespaces).isEmpty ?? true) ? nil : rawInstruct
```
Rebuild: `cd vocal/engine && swift build -c release && cd -`

- [ ] **Step 2: Smoke-test the engine in its new location**

Run from the engine dir (so `default.metallib` is found via cwd):
```bash
cd vocal/engine && echo "This is a smoke test of the consolidated engine." | \
  VOCAL_MODEL_PATH=$(grep '^model_path' ../../vocal.config | cut -d= -f2 | tr -d ' ') \
  VOCAL_SPEAKER=Dylan .build/release/VocalWorker
```
Expected in output: `✅ Model loaded in ...s` then `🗣️ ...s audio in ...s — playing: "This is a smoke test..."`. You should hear it speak. If the model path grep is fragile, hardcode the absolute `VOCAL_MODEL_PATH` for this one check.

- [ ] **Step 3: Build the Rust host and confirm config wires through**

```bash
cargo build
```
Expected: clean build.

- [ ] **Step 4: Manual right-click test (the real acceptance check)**

Run the host: `cargo run` (keep the terminal open). Select text in any app → right-click → "Speak with Vocal". Confirm it speaks. Check `tail -f /tmp/vocal.log` shows `[swift] ✅ Model loaded` and `🗣️ ... playing`. If it fails, the likely cause is `vocal.config` not being found relative to the host's cwd — the `VocalConfig::default()` absolute paths cover this case, so it should still work.

- [ ] **Step 5: Remove dead code**

```bash
git rm -r vocal-engine
git rm main.swift ai_worker.py ai_worker_old.py
```

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: remove dead experiments (vocal-engine, ai_worker, loose main.swift); guard empty instruct"
```

- [ ] **Step 7: Final verification**

```bash
cargo test        # config tests pass
cargo build       # host builds
git status        # clean (vocal.config untracked & ignored; engine artifacts ignored)
git log --oneline # baseline + 4 consolidation commits
```

**Note on the original `../swift-qwen3-tts` repo:** it is left in place as a backup (separate git repo with history). Remove it manually only after you're confident `vocal/engine/` is solid:
```bash
rm -rf ../swift-qwen3-tts   # OPTIONAL — only when you're sure
```

---

## Self-Review (consolidation)

- **Spec coverage:** §3.1 layout (Task 3), §3.2 model external/configurable (Task 2 — absolute paths), §3.3 single-source config (Tasks 1–2), §3.4 dead code + build (Tasks 3–4), §3.5 nothing-breaks smoke test (Task 4). ✓
- **Placeholders:** none; all commands and code are concrete.
- **Type consistency:** `VocalConfig` fields and `engine_binary()`/`engine_cwd()` used consistently in Task 2. ✓
- **Deviation logged:** absolute `vocal.config` paths instead of relative symlink (robustness under macOS Services cwd). ✓
