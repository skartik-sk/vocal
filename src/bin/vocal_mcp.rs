//! Vocal MCP server — exposes Vocal TTS as an agent-callable tool.
//!
//! An AI agent spawns this binary over stdio and calls the `speak` tool to make
//! Vocal speak text aloud on this Mac. The tool loads `vocal.config`, spawns the
//! VocalWorker (streaming, 4-bit), pipes the text to its stdin, and returns once
//! playback finishes. v1 spawns a worker per call; a persistent worker is a
//! future optimization.

use rmcp::{
    handler::server::router::tool::ToolRouter, handler::server::tool::Parameters,
    model::{Implementation, ProtocolVersion, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router, Json, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use vocal::VocalConfig;

/// The MCP service. `tool_router` is populated by the `#[tool_router]` macro.
#[derive(Debug)]
struct VocalMcp {
    tool_router: ToolRouter<VocalMcp>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SpeakArgs {
    #[schemars(description = "The text to speak aloud.")]
    text: String,
    #[schemars(description = "Speaker/voice name, e.g. \"Dylan\". Defaults to vocal.config.")]
    speaker: Option<String>,
    #[schemars(description = "Language, e.g. \"English\". Defaults to vocal.config.")]
    language: Option<String>,
    #[schemars(description = "Style/emotion instruction. Defaults to vocal.config; empty = none.")]
    instruct: Option<String>,
    #[schemars(description = "Sampling temperature 0.0-1.0 as text. Defaults to vocal.config.")]
    temperature: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct SpeakResult {
    ok: bool,
    message: String,
}

#[tool_router]
impl VocalMcp {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "speak",
        description = "Speak the given text aloud on this Mac using local on-device Qwen3-TTS (streaming). Returns when playback finishes."
    )]
    async fn speak(&self, Parameters(args): Parameters<SpeakArgs>) -> Json<SpeakResult> {
        let mut cfg = VocalConfig::load(std::path::Path::new("vocal.config"));
        // Apply per-call overrides (used by the swift backend; ignored by chatterbox).
        if let Some(s) = args.speaker {
            cfg.speaker = s;
        }
        if let Some(l) = args.language {
            cfg.language = l;
        }
        if let Some(i) = args.instruct {
            cfg.instruct = Some(i);
        }
        if let Some(t) = args.temperature {
            cfg.temperature = t;
        }

        // Build the worker Command from the active backend (swift | chatterbox).
        let spec = vocal::worker::launch_spec(&cfg);
        let mut cmd = Command::new(&spec.program);
        cmd.args(&spec.args);
        for (k, v) in &spec.envs {
            cmd.env(k, v);
        }
        if let Some(cwd) = &spec.cwd {
            cmd.current_dir(cwd);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Json(SpeakResult {
                    ok: false,
                    message: format!(
                        "failed to spawn VocalWorker at {}: {e}",
                        cfg.engine_binary().display()
                    ),
                });
            }
        };

        // Feed the text, then close stdin (EOF) so the worker generates, plays, exits.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(args.text.as_bytes()).await;
            let _ = stdin.write_all(b"\n").await;
            // stdin dropped here → pipe closes → EOF.
        }

        let output = match child.wait_with_output().await {
            Ok(o) => o,
            Err(e) => {
                return Json(SpeakResult {
                    ok: false,
                    message: format!("worker did not finish: {e}"),
                });
            }
        };

        if output.status.success() {
            Json(SpeakResult {
                ok: true,
                message: "spoken".to_string(),
            })
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Json(SpeakResult {
                ok: false,
                message: format!(
                    "worker exited {:?}: {}",
                    output.status,
                    stderr.chars().take(400).collect::<String>()
                ),
            })
        }
    }
}

#[tool_handler]
impl ServerHandler for VocalMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "vocal-mcp".to_string(),
                version: "0.1.0".to_string(),
            },
            instructions: Some(
                "Vocal: local on-device text-to-speech. Call the `speak` tool with text to have it spoken aloud on this Mac. Optional args: speaker, language, instruct, temperature.".to_string(),
            ),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let service = VocalMcp::new()
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    service.waiting().await?;
    Ok(())
}
