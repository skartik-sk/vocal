//! Vocal Manager — Tauri backend commands.
//!
//! The manager is a thin shell over the existing Vocal plumbing: it loads
//! `vocal.config` (same parser as the Services host), spawns the same worker
//! subprocesses via `vocal::worker::launch_spec`, and exposes live model-load /
//! speak / unload state to the webview UI over Tauri commands.

pub mod worker;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use vocal::VOCAL_ROOT;

/// Locate `vocal.config`: explicit env override, else the baked project root.
/// The root crate's defaults are absolute, so this works no matter the cwd.
pub fn config_path() -> PathBuf {
    std::env::var_os("VOCAL_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(VOCAL_ROOT).join("vocal.config"))
}

/// Live worker manager shared across commands.
pub struct AppState {
    pub worker: worker::WorkerManager,
    /// Cache of computed model-dir sizes (path → bytes). Walking a multi-hundred
    /// MB model dir on every 300 ms status poll is wasteful, so sizes are
    /// computed once per unique path.
    pub model_sizes: Mutex<std::collections::HashMap<String, u64>>,
}

/// Tauri commands. Kept in their own module so `#[tauri::command]`'s generated
/// `__cmd__*` macros live in a clean namespace.
pub mod commands {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use serde::Serialize;
    use tauri::State;

    use vocal::VocalConfig;
    use vocal::VOCAL_ROOT;

    use crate::config_path;
    use crate::worker::WorkerState;
    use crate::AppState;

    /// Per-backend status for the Manage tab: does the worker binary exist, does
    /// the model dir exist, what's its size, and is it the active backend?
    #[derive(Serialize, Default, Clone)]
    pub struct BackendStatus {
        pub backend: String,
        pub label: String,
        pub active: bool,
        pub binary: String,
        pub binary_exists: bool,
        pub model_path: String,
        pub model_exists: bool,
        pub model_bytes: Option<u64>,
        pub model_note: String,
    }

    /// A Qwen3-TTS checkpoint discovered on disk (0.6B / 1.7B, etc.) so the UI
    /// can list every local model, not just the one in `vocal.config`.
    #[derive(Serialize, Default, Clone)]
    pub struct QwenModel {
        pub path: String,
        pub name: String,
        pub active: bool,
        pub exists: bool,
        pub bytes: Option<u64>,
    }

    #[derive(Serialize, Default)]
    pub struct StatusDto {
        pub config_path: String,
        pub backend: String,
        pub backends: Vec<BackendStatus>,
        pub qwen_models: Vec<QwenModel>,
        pub worker: Option<WorkerState>,
    }

    fn dir_bytes(path: &Path) -> Option<u64> {
        let mut total = 0u64;
        fn walk(p: &Path, total: &mut u64) {
            if let Ok(rd) = fs::read_dir(p) {
                for ent in rd.flatten() {
                    let path = ent.path();
                    if path.is_dir() {
                        walk(&path, total);
                    } else if let Ok(m) = ent.metadata() {
                        *total += m.len();
                    }
                }
            }
        }
        walk(path, &mut total);
        Some(total)
    }

    /// Discover Qwen3-TTS checkpoints next to the configured `model_path` (the
    /// project's model dir holds 0.6B and 1.7B variants). Falls back to just the
    /// configured path. Returns the list sorted by name.
    fn discover_qwen_models(
        cfg: &VocalConfig,
        sizes: &Mutex<HashMap<String, u64>>,
    ) -> Vec<QwenModel> {
        fn build(
            p: &Path,
            cfg_model_path: &str,
            sizes: &Mutex<HashMap<String, u64>>,
        ) -> QwenModel {
            let path = p.to_string_lossy().into_owned();
            let name = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone());
            let exists = p.is_dir();
            let bytes = if exists {
                let key = path.clone();
                let cached = sizes.lock().unwrap().get(&key).copied();
                cached.or_else(|| {
                    let v = dir_bytes(p);
                    if let Some(v) = v {
                        sizes.lock().unwrap().insert(key, v);
                    }
                    v
                })
            } else {
                None
            };
            let active = path == cfg_model_path;
            QwenModel {
                path,
                name,
                active,
                exists,
                bytes,
            }
        }

        let mut found = Vec::new();
        let configured = PathBuf::from(&cfg.model_path);
        // Scan the parent dir for sibling Qwen checkpoints (e.g. ~/Developer/fun-projects).
        if let Some(parent) = configured.parent() {
            if let Ok(rd) = fs::read_dir(parent) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    let name = p.file_name().map(|s| s.to_string_lossy().into_owned());
                    if p.is_dir()
                        && name
                            .as_deref()
                            .is_some_and(|n| n.contains("Qwen3-TTS") && n.ends_with("-4bit"))
                    {
                        found.push(build(&p, &cfg.model_path, sizes));
                    }
                }
            }
        }
        // Ensure the configured path is always present even if the scan missed it.
        if !found.iter().any(|m| m.path == cfg.model_path) {
            found.push(build(&configured, &cfg.model_path, sizes));
        }
        found.sort_by(|a, b| a.name.cmp(&b.name));
        found
    }

    /// Build the status card for one backend from the config. `sizes` is the
    /// shared model-size cache (path → bytes) so we don't re-walk huge dirs.
    fn backend_status(
        cfg: &VocalConfig,
        backend: &str,
        sizes: &Mutex<HashMap<String, u64>>,
    ) -> BackendStatus {
        let mut s = BackendStatus {
            backend: backend.to_string(),
            active: cfg.backend == backend,
            ..Default::default()
        };
        let size_of = |p: &Path| -> Option<u64> {
            let key = p.to_string_lossy().into_owned();
            if let Some(v) = sizes.lock().unwrap().get(&key) {
                return Some(*v);
            }
            let v = dir_bytes(p)?;
            sizes.lock().unwrap().insert(key, v);
            Some(v)
        };
        match backend {
            "swift" => {
                s.label = "Qwen3 (Swift)".into();
                s.binary = cfg.engine_binary().to_string_lossy().into_owned();
                s.binary_exists = cfg.engine_binary().is_file();
                s.model_path = cfg.model_path.clone();
                let p = Path::new(&cfg.model_path);
                s.model_exists = p.is_dir();
                s.model_bytes = s.model_exists.then(|| size_of(p)).flatten();
                s.model_note = format!("{} speakers", 12);
            }
            "chatterbox" => {
                s.label = "Chatterbox (Python)".into();
                s.binary = cfg.python_bin.clone();
                s.binary_exists = Path::new(&cfg.python_bin).is_file();
                s.model_path = cfg.chatterbox_model.clone();
                // HF repo id may not be a local dir — note that gracefully.
                let p = Path::new(&cfg.chatterbox_model);
                s.model_exists = p.is_dir();
                s.model_bytes = s.model_exists.then(|| size_of(p)).flatten();
                s.model_note = if s.model_exists {
                    String::new()
                } else {
                    "HF repo id — downloaded on first load".into()
                };
            }
            _ => {
                // native_chatterbox
                s.label = "Chatterbox (Native)".into();
                s.binary = cfg.chatterbox_binary().to_string_lossy().into_owned();
                s.binary_exists = cfg.chatterbox_binary().is_file();
                s.model_path = cfg.chatterbox_model_path.clone();
                let p = Path::new(&cfg.chatterbox_model_path);
                s.model_exists = p.is_dir();
                s.model_bytes = s.model_exists.then(|| size_of(p)).flatten();
                s.model_note = "⚠️ Experimental — the full pipeline runs (T3→S3→vocoder) but the output is still being debugged (noise). Use Chatterbox (Python) for clean audio."
                    .into();
            }
        }
        s
    }

    /// Load config + build full status for the frontend.
    #[tauri::command]
    pub fn get_config() -> Result<serde_json::Value, String> {
        let path = config_path();
        let cfg = VocalConfig::load(&path);
        serde_json::to_value(cfg).map_err(|e| e.to_string())
    }

    #[tauri::command]
    pub fn save_config(cfg: VocalConfig) -> Result<(), String> {
        let path = config_path();
        fs::write(&path, cfg.serialize())
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    /// Point the Qwen (swift) backend at a different local checkpoint, e.g. the
    /// 1.7B model, and persist the change to vocal.config.
    #[tauri::command]
    pub fn set_qwen_model(model_path: String) -> Result<(), String> {
        let path = config_path();
        let mut cfg = VocalConfig::load(&path);
        cfg.backend = "swift".into();
        cfg.model_path = model_path;
        fs::write(&path, cfg.serialize())
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    #[tauri::command]
    pub fn check_status(state: State<AppState>) -> Result<serde_json::Value, String> {
        let path = config_path();
        let cfg = VocalConfig::load(&path);
        let mut dto = StatusDto {
            config_path: path.to_string_lossy().into_owned(),
            backend: cfg.backend.clone(),
            worker: Some(state.worker.state()),
            ..Default::default()
        };
        for b in ["swift", "chatterbox", "native_chatterbox"] {
            dto.backends.push(backend_status(&cfg, b, &state.model_sizes));
        }
        dto.qwen_models = discover_qwen_models(&cfg, &state.model_sizes);
        serde_json::to_value(dto).map_err(|e| e.to_string())
    }

    #[tauri::command]
    pub fn worker_start(state: State<AppState>) -> Result<(), String> {
        let path = config_path();
        let cfg = VocalConfig::load(&path);
        state.worker.start(&cfg)
    }

    #[tauri::command]
    pub fn worker_speak(state: State<AppState>, text: String) -> Result<(), String> {
        state.worker.speak(&text)
    }

    #[tauri::command]
    pub fn worker_stop(state: State<AppState>) -> Result<(), String> {
        state.worker.stop()
    }

    #[tauri::command]
    pub fn worker_state(state: State<AppState>) -> Result<serde_json::Value, String> {
        serde_json::to_value(state.worker.state()).map_err(|e| e.to_string())
    }

    #[tauri::command]
    pub fn worker_logs(state: State<AppState>, offset: usize) -> Result<serde_json::Value, String> {
        serde_json::to_value(state.worker.logs(offset)).map_err(|e| e.to_string())
    }

    // ── MCP server control ──────────────────────────────────────────────
    // `target/release/vocal_mcp` is a stdio MCP server an AI client (Claude Code, …)
    // spawns to call Vocal's `speak` tool. These commands surface its status, install the
    // client config into `.mcp.json`, and smoke-test the JSON-RPC handshake from the UI.

    fn mcp_binary_path() -> PathBuf {
        Path::new(VOCAL_ROOT).join("target/release/vocal_mcp")
    }

    #[derive(Serialize)]
    pub struct McpStatus {
        pub binary: String,
        pub binary_exists: bool,
        pub config_json: String,
        pub config_path: String,
        pub config_installed: bool,
    }

    #[tauri::command]
    pub fn mcp_status() -> Result<serde_json::Value, String> {
        let bin = mcp_binary_path();
        let cfg_path = Path::new(VOCAL_ROOT).join(".mcp.json");
        let config_json = serde_json::json!({
            "mcpServers": { "vocal": { "command": bin.to_string_lossy() } }
        })
        .to_string();
        serde_json::to_value(McpStatus {
            binary: bin.to_string_lossy().into_owned(),
            binary_exists: bin.is_file(),
            config_json,
            config_path: cfg_path.to_string_lossy().into_owned(),
            config_installed: cfg_path.is_file(),
        })
        .map_err(|e| e.to_string())
    }

    /// Write/merge the vocal entry into the project `.mcp.json` so an MCP client picks it up.
    #[tauri::command]
    pub fn mcp_install_config() -> Result<String, String> {
        let bin = mcp_binary_path();
        if !bin.is_file() {
            return Err(format!(
                "vocal_mcp not built — run `cargo build --release` in {VOCAL_ROOT}"
            ));
        }
        let cfg_path = Path::new(VOCAL_ROOT).join(".mcp.json");
        let mut doc: serde_json::Value = std::fs::read_to_string(&cfg_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::json!({}));
        if doc.get("mcpServers").is_none() {
            doc["mcpServers"] = serde_json::json!({});
        }
        doc["mcpServers"]["vocal"] = serde_json::json!({ "command": bin.to_string_lossy() });
        let pretty = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
        std::fs::write(&cfg_path, &pretty)
            .map_err(|e| format!("write {}: {e}", cfg_path.display()))?;
        Ok(cfg_path.to_string_lossy().into_owned())
    }

    /// Spawn vocal_mcp, run initialize + tools/list, return the tool names — proves the
    /// server is alive and reachable.
    #[tauri::command]
    pub async fn mcp_test() -> Result<serde_json::Value, String> {
        use std::time::Duration;
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::process::Command;
        use tokio::time::timeout;

        let bin = mcp_binary_path();
        if !bin.is_file() {
            return Err("vocal_mcp not built".into());
        }
        let mut child = Command::new(&bin)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn failed: {e}"))?;
        let mut stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;

        let init = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"vocal-manager\",\"version\":\"0.1.0\"}}}\n";
        stdin.write_all(init.as_bytes()).await.map_err(|e| e.to_string())?;
        stdin
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await
            .map_err(|e| e.to_string())?;
        stdin
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n")
            .await
            .map_err(|e| e.to_string())?;
        drop(stdin);

        let mut lines = BufReader::new(stdout).lines();
        let mut tools: Vec<String> = Vec::new();
        let mut server_name = String::new();
        let mut init_ok = false;
        while let Ok(Ok(Some(line))) = timeout(Duration::from_secs(8), lines.next_line()).await {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            if v.get("id") == Some(&serde_json::json!(1)) {
                init_ok = true;
                if let Some(n) = v.pointer("/result/serverInfo/name").and_then(|x| x.as_str()) {
                    server_name = n.into();
                }
            }
            if let Some(arr) = v.pointer("/result/tools").and_then(|x| x.as_array()) {
                for t in arr {
                    if let Some(n) = t.get("name").and_then(|x| x.as_str()) {
                        tools.push(n.into());
                    }
                }
                break;
            }
        }
        let _ = child.kill().await;
        if !init_ok {
            return Err("MCP server did not complete initialize (timeout)".into());
        }
        Ok(serde_json::json!({ "server": server_name, "tools": tools }))
    }
}
