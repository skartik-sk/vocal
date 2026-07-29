use std::fs;
use std::path::{Path, PathBuf};

/// Runtime configuration for Vocal. Loaded from a gitignored `vocal.config`
/// (`key = value` lines); falls back to baked-in defaults so the app works on
/// the dev machine even without the file. Rust is the single source of truth —
/// it passes these to the Swift worker via env vars (see main.rs).
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
            instruct: Some(
                "be very Fast, Serious, and not skip any word like you are reading audiobook".into(),
            ),
            temperature: "0.8".into(),
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

    /// Path to the compiled VocalWorker binary: `<engine_dir>/.build/release/VocalWorker`.
    pub fn engine_binary(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir).join(".build/release/VocalWorker")
    }

    /// Working directory for the worker (so `default.metallib` is found via cwd).
    pub fn engine_cwd(&self) -> PathBuf {
        PathBuf::from(&self.engine_dir)
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
        assert_eq!(
            cfg.engine_binary(),
            PathBuf::from("/tmp/engine/.build/release/VocalWorker")
        );
        assert_eq!(cfg.engine_cwd(), PathBuf::from("/tmp/engine"));
    }
}
