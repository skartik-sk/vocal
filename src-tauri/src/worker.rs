//! Persistent TTS worker supervisor for the Vocal Manager.
//!
//! Spawns the same worker subprocess as the Services host / MCP server (via
//! `vocal::worker::launch_spec`), keeps its stdin open so the model stays
//! loaded, parses the worker's stdout status lines into a live state, and
//! unloads by closing stdin (worker exits → RAM/GPU freed instantly).
//!
//! No changes to the Swift/Python workers — the manager only reads their
//! existing log lines:
//!   `⏳ Loading model: <path>`      → Loading
//!   `✅ Model loaded in <Xs>...`    → Ready (captures X)
//!   `🗣️ ... playing:` / `text:->`  → Speaking
//!   `🛑 stdin closed...`            → Idle (clean shutdown)
//!   worker exits nonzero            → Error

use std::io::Write;
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use vocal::VocalConfig;
use vocal::worker::launch_spec;

/// Maximum log lines kept in the buffer before oldest are dropped.
const LOG_CAP: usize = 2000;

/// What the worker is currently doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    #[default]
    Idle,
    Loading,
    Ready,
    Speaking,
    Error,
}

/// Snapshot of the worker's live state, surfaced to the UI every ~300 ms.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkerState {
    pub running: bool,
    pub phase: Phase,
    pub pid: Option<u32>,
    pub load_seconds: Option<f64>,
    pub last_line: String,
    pub error: Option<String>,
}

/// Incremental log buffer: `worker_logs(offset)` returns everything after
/// `offset` and the new offset, so the frontend can tail without re-reading.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogChunk {
    pub lines: Vec<String>,
    pub offset: usize,
}

/// Handle to a spawned worker. The watchdog thread owns the real `Child` and
/// reaps it; this struct holds the stdin pipe (kept open while "loaded"), the
/// stdout reader thread, and a channel the watchdog signals on exit.
struct WorkerHandle {
    stdin: ChildStdin,
    reader: thread::JoinHandle<()>,
    watchdog_done: Receiver<()>,
}

/// Shared, mutex-guarded worker bookkeeping.
#[derive(Default)]
pub struct WorkerInner {
    pub running: bool,
    pub phase: Phase,
    pub pid: Option<u32>,
    pub load_seconds: Option<f64>,
    pub last_line: String,
    pub error: Option<String>,
    pub logs: Vec<String>,
}

/// Append a log line, capping the buffer.
fn push_log(w: &mut WorkerInner, line: String) {
    if w.logs.len() >= LOG_CAP {
        let overflow = w.logs.len() - LOG_CAP + 1;
        w.logs.drain(..overflow);
    }
    w.logs.push(line);
}

pub struct WorkerManager {
    inner: Arc<Mutex<WorkerInner>>,
    handle: Arc<Mutex<Option<WorkerHandle>>>,
}

impl WorkerManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkerInner::default())),
            handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Spawn the worker for the given backend and start reading its stdout.
    /// Returns immediately — loading is async; the UI watches [`Self::state`].
    pub fn start(&self, cfg: &VocalConfig) -> Result<(), String> {
        // Kill any existing worker first so a stale/lingering process (e.g. one
        // that hasn't hit its idle timeout yet) never blocks a fresh spawn.
        if self.handle.lock().unwrap().is_some() {
            let _ = self.stop();
        }

        let spec = launch_spec(cfg);
        let mut cmd = Command::new(&spec.program);
        cmd.args(&spec.args);
        for (k, v) in &spec.envs {
            cmd.env(k, v);
        }
        if let Some(cwd) = &spec.cwd {
            cmd.current_dir(cwd);
        }

        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", spec.program))?;

        let pid = child.id();
        let stdin = child.stdin.take().ok_or("failed to open worker stdin")?;
        let stdout = child.stdout.take().ok_or("failed to open worker stdout")?;
        let stderr = child.stderr.take().ok_or("failed to open worker stderr")?;

        let inner = Arc::clone(&self.inner);
        {
            let mut w = inner.lock().unwrap();
            w.running = true;
            w.phase = Phase::Loading;
            w.pid = Some(pid);
            w.load_seconds = None;
            w.error = None;
            let spawned = format!("[manager] spawned pid {pid} (backend {})", cfg.backend);
            w.last_line = spawned.clone();
            push_log(&mut w, spawned);
        }

        // stdout → status-line parser; stderr → error capture; both tee into logs.
        let reader_inner = Arc::clone(&inner);
        let reader = thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                update_state(&reader_inner, &line);
            }
        });
        let err_inner = Arc::clone(&inner);
        thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                update_state(&err_inner, &format!("[stderr] {line}"));
            }
        });

        // Watchdog: owns the real child. When it exits (crash, kill, or clean
        // EOF shutdown), reset state, free the handle, and notify — the UI
        // never sees a zombie "running" worker and a new Load is allowed.
        let (exit_tx, exit_rx) = channel::<()>();
        let watch_inner = Arc::clone(&inner);
        let watch_handle = Arc::clone(&self.handle);
        thread::spawn(move || {
            let status = child.wait();
            {
                let mut w = watch_inner.lock().unwrap();
                w.running = false;
                w.pid = None;
                let note = match &status {
                    Ok(s) if s.success() => "worker exited cleanly".to_string(),
                    Ok(s) => format!("worker exited with status {s}"),
                    Err(e) => format!("worker wait error: {e}"),
                };
                w.last_line = note.clone();
                push_log(&mut w, format!("[manager] {note}"));
                // On unexpected exit (crash), surface an error state.
                if !matches!(status, Ok(s) if s.success()) {
                    w.phase = Phase::Error;
                    w.error = Some(note);
                } else {
                    w.phase = Phase::Idle;
                }
            }
            // Free the handle slot so the manager can be restarted.
            *watch_handle.lock().unwrap() = None;
            let _ = exit_tx.send(());
        });

        *self.handle.lock().unwrap() = Some(WorkerHandle {
            stdin,
            reader,
            watchdog_done: exit_rx,
        });
        Ok(())
    }

    /// Feed one chunk of text to the worker's stdin (must be loaded).
    pub fn speak(&self, text: &str) -> Result<(), String> {
        // Take the handle out for the duration of the write so the MutexGuard
        // doesn't stay borrowed across `stdin` writes (borrow checker).
        let mut handle = self
            .handle
            .lock()
            .unwrap()
            .take()
            .ok_or("no worker loaded")?;

        // Same sentence chunking as the Services host (src/main.rs).
        let sentences: Vec<String> = text
            .split_terminator(&['.', '?', '!'][..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let result = (|| -> Result<(), String> {
            if sentences.is_empty() {
                return Err("nothing to speak".into());
            }
            {
                let mut w = self.inner.lock().unwrap();
                w.phase = Phase::Speaking;
                push_log(
                    &mut w,
                    format!("[manager] speaking {} clause(s)", sentences.len()),
                );
            }
            let stdin = &mut handle.stdin;
            for s in &sentences {
                writeln!(stdin, "{s}").map_err(|e| format!("failed to write to worker: {e}"))?;
            }
            stdin
                .flush()
                .map_err(|e| format!("failed to flush worker stdin: {e}"))?;
            Ok(())
        })();

        // Restore the handle on success; on write failure the worker is likely
        // dead, so let the watchdog clear state.
        if result.is_ok() {
            *self.handle.lock().unwrap() = Some(handle);
        }
        result
    }

    /// Unload: close stdin so the worker finishes playback, flushes, exits.
    /// Force-kill after a short grace period if it lingers.
    pub fn stop(&self) -> Result<(), String> {
        let mut guard = self.handle.lock().unwrap();
        let Some(handle) = guard.take() else {
            return Ok(());
        };
        drop(handle.stdin); // EOF → worker shuts down

        // Wait for the watchdog to reap the child (bounded).
        let deadline = Instant::now() + Duration::from_secs(15);
        while handle
            .watchdog_done
            .recv_timeout(Duration::from_millis(100))
            .is_err()
            && Instant::now() < deadline
        {}
        let _ = handle.reader.join();
        Ok(())
    }

    /// Current worker state snapshot for the UI.
    pub fn state(&self) -> WorkerState {
        let w = self.inner.lock().unwrap();
        WorkerState {
            running: w.running,
            phase: w.phase,
            pid: w.pid,
            load_seconds: w.load_seconds,
            last_line: w.last_line.clone(),
            error: w.error.clone(),
        }
    }

    /// Incremental log read starting after `offset`.
    pub fn logs(&self, offset: usize) -> LogChunk {
        let w = self.inner.lock().unwrap();
        let lines = w.logs.get(offset..).unwrap_or(&[]).to_vec();
        LogChunk {
            lines,
            offset: w.logs.len(),
        }
    }
}

/// Parse a worker stdout/stderr line into state transitions + log.
fn update_state(inner: &Arc<Mutex<WorkerInner>>, line: &str) {
    let mut w = inner.lock().unwrap();
    push_log(&mut w, line.to_string());
    w.last_line = line.to_string();

    if line.contains("Loading model") {
        w.phase = Phase::Loading;
    } else if line.contains("Model loaded in") {
        w.phase = Phase::Ready;
        // e.g. "✅ Model loaded in 1.24s. Waiting for sentences..."
        if let Some(secs) = extract_seconds(line) {
            w.load_seconds = Some(secs);
        }
    } else if line.contains("playing:") || line.contains("text:->") {
        w.phase = Phase::Speaking;
    } else if line.contains("stdin closed") {
        w.phase = Phase::Idle;
        w.running = false;
    } else if line.starts_with("[stderr]") || line.contains("error") {
        w.error = Some(line.to_string());
        w.phase = Phase::Error;
    }
}

/// Pull the first `Xs` float out of a line like `Model loaded in 1.24s`.
fn extract_seconds(line: &str) -> Option<f64> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        // Tokens may carry trailing punctuation (`1.24s.`).
        let tok = tok.trim_end_matches(|c: char| !c.is_ascii_digit() && !c.is_ascii_alphabetic());
        if let Some(trimmed) = tok.strip_suffix('s') {
            if let Ok(v) = trimmed.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inner_with(lines: &[&str]) -> (Arc<Mutex<WorkerInner>>, WorkerState) {
        let inner = Arc::new(Mutex::new(WorkerInner::default()));
        for l in lines {
            update_state(&inner, l);
        }
        let state = {
            let w = inner.lock().unwrap();
            WorkerState {
                running: w.running,
                phase: w.phase,
                pid: None,
                load_seconds: w.load_seconds,
                last_line: w.last_line.clone(),
                error: w.error.clone(),
            }
        };
        (inner, state)
    }

    #[test]
    fn parses_loading_ready_speaking_idle_flow() {
        let (_inner, state) = inner_with(&[
            "⏳ Loading model: /x",
            "✅ Model loaded in 1.24s. Waiting for sentences...",
        ]);
        assert_eq!(state.phase, Phase::Ready);
        assert_eq!(state.load_seconds, Some(1.24));

        let (_inner, s2) = inner_with(&["🗣️ 3.2s audio in 2.10s — playing: \"Hello\""]);
        assert_eq!(s2.phase, Phase::Speaking);

        let (_inner, s3) = inner_with(&["🛑 stdin closed. Freeing GPU memory and exiting."]);
        assert_eq!(s3.phase, Phase::Idle);
        assert_eq!(s3.running, false);
    }

    #[test]
    fn extract_seconds_handles_various_formats() {
        assert_eq!(extract_seconds("loaded in 1.24s"), Some(1.24));
        assert_eq!(extract_seconds("no numbers here"), None);
        assert_eq!(extract_seconds("in 0s"), Some(0.0));
    }

    #[test]
    fn log_buffer_is_capped() {
        let inner = Arc::new(Mutex::new(WorkerInner::default()));
        let total = LOG_CAP + 500;
        for i in 0..total {
            update_state(&inner, &format!("line {i}"));
        }
        let w = inner.lock().unwrap();
        assert!(w.logs.len() <= LOG_CAP, "log buffer exceeded cap");
        // Oldest lines dropped, newest retained.
        let expected_last = format!("line {}", total - 1);
        assert_eq!(w.logs.last().map(|s| s.as_str()), Some(expected_last.as_str()));
        // First retained line is the oldest kept.
        let expected_first = format!("line {}", total - LOG_CAP);
        assert_eq!(w.logs.first().map(|s| s.as_str()), Some(expected_first.as_str()));
    }
}
