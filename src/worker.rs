//! Backend-agnostic TTS worker launch spec.
//!
//! Both the macOS Services host (`src/main.rs`, sync) and the MCP server
//! (`src/bin/vocal_mcp.rs`, async/tokio) build their `Command` from this, so the
//! backend choice (Swift Qwen3 / Python Chatterbox / native Swift Chatterbox) lives
//! in exactly one place.

use crate::config::VocalConfig;
use std::path::PathBuf;

/// Everything needed to spawn the worker process, independent of which async
/// runtime the caller uses. Callers set stdin/stdout/stderr themselves.
pub struct WorkerLaunch {
    pub program: String,
    pub args: Vec<String>,
    pub envs: Vec<(String, String)>,
    pub cwd: Option<PathBuf>,
}

/// Build the launch spec from the active backend in `cfg`.
pub fn launch_spec(cfg: &VocalConfig) -> WorkerLaunch {
    match cfg.backend.as_str() {
        "native_chatterbox" => {
            // Pure-Swift multilingual Chatterbox worker (no Python). Loads the
            // chatterbox-4bit model natively and speaks Hindi + English.
            WorkerLaunch {
                program: cfg.chatterbox_binary().to_string_lossy().into_owned(),
                args: vec![],
                envs: vec![
                    (
                        "CHATTERBOX_ML_MODEL".into(),
                        cfg.chatterbox_model_path.clone(),
                    ),
                    (
                        "CHATTERBOX_ML_LANG".into(),
                        cfg.language.clone(),
                    ),
                    (
                        "CHATTERBOX_ML_MAX_TOKENS".into(),
                        "300".into(),
                    ),
                    (
                        "CHATTERBOX_ML_MEM_MB".into(),
                        "1024".into(),
                    ),
                ],
                cwd: Some(cfg.engine_cwd()),
            }
        }
        "chatterbox" => {
            let mut envs = vec![("CHATTERBOX_MODEL".into(), cfg.chatterbox_model.clone())];
            if let Some(ref r) = cfg.ref_audio {
                envs.push(("CHATTERBOX_REF_AUDIO".into(), r.clone()));
            }
            WorkerLaunch {
                program: cfg.python_bin.clone(),
                args: vec![cfg.chatterbox_worker.clone()],
                envs,
                cwd: None,
            }
        }
        _ => {
            // Swift Qwen3-TTS VocalWorker.
            WorkerLaunch {
                program: cfg.engine_binary().to_string_lossy().into_owned(),
                args: vec![],
                envs: vec![
                    ("VOCAL_MODEL_PATH".into(), cfg.model_path.clone()),
                    ("VOCAL_SPEAKER".into(), cfg.speaker.clone()),
                    ("VOCAL_LANGUAGE".into(), cfg.language.clone()),
                    ("VOCAL_INSTRUCT".into(), cfg.instruct.clone().unwrap_or_default()),
                    ("VOCAL_TEMPERATURE".into(), cfg.temperature.clone()),
                ],
                cwd: Some(cfg.engine_cwd()),
            }
        }
    }
}
