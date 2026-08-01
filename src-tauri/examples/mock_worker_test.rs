//! Ad-hoc integration check for the WorkerManager against a mock worker.
//! Run: cargo run --example mock_worker_test
//! The mock worker is written to a temp file at runtime and emits the same
//! status lines as the real Swift/Python workers, so this validates the parse
//! pipeline without loading a real model.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::thread;
use std::time::Duration;

use vocal::VocalConfig;
use vocal_manager_lib::worker::{Phase, WorkerManager};

/// Write a mock worker script to a temp path and return it.
fn make_mock_worker(mode: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("mock_worker_{mode}.sh"));
    let body = match mode {
        "clean" => r#"#!/bin/bash
echo "⏳ Loading model: /mock"
sleep 0.5
echo "✅ Model loaded in 1.24s. Waiting for sentences..."
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "🗣️ 2.0s audio in 1.00s — playing: \"$line\""
done
echo "🛑 stdin closed. Freeing GPU memory and exiting."
"#,
        // Exits nonzero immediately — models a crash (e.g. bad model path).
        "crash" => r#"#!/bin/bash
echo "⏳ Loading model: /nonexistent"
echo "❌ failed to load model"
exit 1
"#,
        _ => unreachable!(),
    };
    fs::write(&path, body).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn cfg_for(mock: &std::path::Path) -> VocalConfig {
    let mut cfg = VocalConfig::default();
    cfg.backend = "chatterbox".into();
    cfg.python_bin = mock.to_string_lossy().into_owned();
    cfg.chatterbox_worker = "".into();
    cfg
}

fn main() {
    // --- happy path: load → ready → speak → stop ---
    let clean = make_mock_worker("clean");
    let mgr = WorkerManager::new();

    println!("== start ==");
    mgr.start(&cfg_for(&clean)).expect("start failed");

    let mut phase = Phase::Idle;
    for _ in 0..40 {
        thread::sleep(Duration::from_millis(100));
        let s = mgr.state();
        phase = s.phase;
        if s.phase == Phase::Ready {
            println!("   phase=Ready load_seconds={:?}", s.load_seconds);
            break;
        }
    }
    assert_eq!(phase, Phase::Ready, "expected Ready, got {phase:?}");

    println!("== speak ==");
    mgr.speak("Hello there. This is a test!").expect("speak failed");
    thread::sleep(Duration::from_millis(300));
    let s = mgr.state();
    println!("   phase after speak: {:?}", s.phase);
    assert_eq!(s.phase, Phase::Speaking);

    println!("== stop ==");
    mgr.stop().expect("stop failed");
    let s = mgr.state();
    println!("   final phase: {:?} running: {}", s.phase, s.running);
    assert!(!s.running, "worker should not be running after stop");

    // --- crash path: worker dies → state resets to Error, restartable ---
    let crash = make_mock_worker("crash");
    let mgr2 = WorkerManager::new();
    mgr2.start(&cfg_for(&crash)).expect("start (crash) failed");

    let mut crashed = false;
    for _ in 0..40 {
        thread::sleep(Duration::from_millis(100));
        if !mgr2.state().running {
            crashed = true;
            break;
        }
    }
    assert!(crashed, "crashing worker never reset running state");
    let s = mgr2.state();
    println!("== crash ==\n   phase: {:?} running: {}", s.phase, s.running);
    // The handle should be freed so a new start is allowed.
    mgr2.start(&cfg_for(&clean)).expect("restart after crash failed");
    mgr2.stop().expect("stop after restart failed");
    println!("   restart-after-crash OK");

    let _ = fs::remove_file(&clean);
    let _ = fs::remove_file(&crash);
    println!("MOCK TEST PASSED");
}
