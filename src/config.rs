use std::fs;
use std::path::{Path, PathBuf};

/// The project root, resolved at compile time to this crate's directory. Used by
/// [`VocalConfig::default`] and the Tauri manager to locate `vocal.config`,
/// `target/release/vocal_mcp`, etc. regardless of process cwd (macOS Services
/// launches with an unpredictable cwd).
pub const VOCAL_ROOT: &str = env!("CARGO_MANIFEST_DIR");

/// Runtime configuration for Vocal. Loaded from a gitignored `vocal.config`
/// (`key = value` lines); falls back to baked-in defaults. Rust is the single
/// source of truth — it builds the worker launch spec (see `worker.rs`) and
/// passes settings to the worker via env vars.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VocalConfig {
    /// `"swift"` (Qwen3-TTS via the Swift engine), `"chatterbox"` (Python mlx-audio),
    /// or `"native_chatterbox"` (pure-Swift Chatterbox, no Python).
    pub backend: String,

    // --- swift / Qwen3-TTS backend ---
    pub model_path: String,
    pub engine_dir: String,
    pub speaker: String,
    pub language: String,
    pub instruct: Option<String>,
    pub temperature: String,

    // --- chatterbox / mlx-audio backend ---
    pub python_bin: String,
    pub chatterbox_worker: String,
    pub chatterbox_model: String,
    pub ref_audio: Option<String>,

    // --- native_chatterbox backend (pure Swift, no Python) ---
    /// Local model dir holding config.json + model.safetensors + conds.safetensors.
    pub chatterbox_model_path: String,
}

impl Default for VocalConfig {
    fn default() -> Self {
        let home = VOCAL_ROOT;
        Self {
            backend: "native_chatterbox".into(),
            // No baked default — set `model_path` in vocal.config (see vocal.config.example).
            model_path: String::new(),
            engine_dir: format!("{home}/engine"),
            speaker: "Dylan".into(),
            language: "hi".into(),
            instruct: Some(
                "be very Fast, Serious, and not skip any word like you are reading audiobook".into(),
            ),
            temperature: "0.8".into(),
            python_bin: format!("{home}/src/.venv/bin/python"),
            chatterbox_worker: format!("{home}/scripts/chatterbox_worker.py"),
            chatterbox_model: "mlx-community/chatterbox-turbo-4bit".into(),
            ref_audio: None,
            // No baked default — set `chatterbox_model_path` in vocal.config (HF cache snapshot).
            chatterbox_model_path: String::new(),
        }
    }
}

impl VocalConfig {
    /// Load `vocal.config`, applying any present keys over the defaults. If the
    /// file is missing or unreadable, returns the defaults unchanged.
    pub fn load(path: &Path) -> VocalConfig {
        let mut cfg = VocalConfig::default();
        if let Ok(text) = fs::read_to_string(path) {
            for (key, value) in parse_kv(&text) {
                match key.as_str() {
                    "backend" => cfg.backend = value,
                    "model_path" => cfg.model_path = value,
                    "engine_dir" => cfg.engine_dir = value,
                    "speaker" => cfg.speaker = value,
                    "language" => cfg.language = value,
                    "instruct" => cfg.instruct = if value.is_empty() { None } else { Some(value) },
                    "temperature" => cfg.temperature = value,
                    "python_bin" => cfg.python_bin = value,
                    "chatterbox_worker" => cfg.chatterbox_worker = value,
                    "chatterbox_model" => cfg.chatterbox_model = value,
                    "ref_audio" => cfg.ref_audio = if value.is_empty() { None } else { Some(value) },
                    "chatterbox_model_path" => cfg.chatterbox_model_path = value,
                    _ => {} // ignore unknown keys (forward-compatible)
                }
            }
        }
        cfg
    }

    /// Path to the compiled Swift VocalWorker binary: `<engine_dir>/.build/release/VocalWorker`.
    pub fn engine_binary(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir).join(".build/release/VocalWorker")
    }

    /// Working directory for the Swift worker (so `default.metallib` is found via cwd).
    pub fn engine_cwd(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir)
    }

    /// Path to the compiled native Chatterbox worker: `<engine_dir>/.build/release/ChatterboxMLWorker`.
    pub fn chatterbox_binary(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir).join(".build/release/ChatterboxMLWorker")
    }

    /// Serialize this config back to `key = value` lines — the mirror of
    /// [`parse_kv`]. Lets the Tauri manager persist edits made in its UI.
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        let mut kv = |k: &str, v: &str| {
            out.push_str(k);
            out.push_str(" = ");
            out.push_str(v);
            out.push('\n');
        };
        kv("backend", &self.backend);
        kv("model_path", &self.model_path);
        kv("engine_dir", &self.engine_dir);
        kv("speaker", &self.speaker);
        kv("language", &self.language);
        kv("instruct", self.instruct.as_deref().unwrap_or(""));
        kv("temperature", &self.temperature);
        kv("python_bin", &self.python_bin);
        kv("chatterbox_worker", &self.chatterbox_worker);
        kv("chatterbox_model", &self.chatterbox_model);
        kv("ref_audio", self.ref_audio.as_deref().unwrap_or(""));
        kv("chatterbox_model_path", &self.chatterbox_model_path);
        out
    }
}

/// Parse `key = value` lines: trims, skips blanks and `#` comments, ignores
/// lines without `=`. Values are taken verbatim (already trimmed).
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
        let none = VocalConfig::load(Path::new("/does/not/exist/vocal.config"));
        assert_eq!(none.speaker, "Dylan");
        assert_eq!(none.backend, "native_chatterbox");

        let p = std::env::temp_dir().join("vocal_config_test.cfg");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            writeln!(f, "speaker = Aiden").unwrap();
            writeln!(f, "backend = native_chatterbox").unwrap();
            writeln!(f, "instruct =").unwrap();
        }
        let cfg = VocalConfig::load(&p);
        assert_eq!(cfg.speaker, "Aiden");
        assert_eq!(cfg.backend, "native_chatterbox");
        assert_eq!(cfg.instruct, None);
        assert_eq!(cfg.temperature, "0.8");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn engine_binary_and_cwd_derive_from_engine_dir() {
        let mut cfg = VocalConfig::default();
        cfg.engine_dir = "/tmp/engine".into();
        assert_eq!(
            cfg.engine_binary(),
            PathBuf::from("/tmp/engine/.build/release/VocalWorker")
        );
        assert_eq!(
            cfg.chatterbox_binary(),
            PathBuf::from("/tmp/engine/.build/release/ChatterboxMLWorker")
        );
        assert_eq!(cfg.engine_cwd(), PathBuf::from("/tmp/engine"));
    }

    #[test]
    fn serialize_round_trips_through_load() {
        let mut cfg = VocalConfig::default();
        cfg.speaker = "Aiden".into();
        cfg.temperature = "0.7".into();
        cfg.instruct = None; // must serialize as empty line → reload as None
        cfg.ref_audio = Some("/tmp/voice.wav".into());

        let p = std::env::temp_dir().join("vocal_config_roundtrip.cfg");
        std::fs::write(&p, cfg.serialize()).unwrap();
        let loaded = VocalConfig::load(&p);
        let _ = std::fs::remove_file(&p);

        assert_eq!(loaded.backend, cfg.backend);
        assert_eq!(loaded.speaker, "Aiden");
        assert_eq!(loaded.temperature, "0.7");
        assert_eq!(loaded.instruct, None);
        assert_eq!(loaded.ref_audio.as_deref(), Some("/tmp/voice.wav"));
        assert_eq!(loaded.chatterbox_model_path, cfg.chatterbox_model_path);
    }
}
