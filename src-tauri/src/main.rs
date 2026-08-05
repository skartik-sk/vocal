//! Vocal Manager — Tauri entry point.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::sync::Mutex;

use vocal_manager_lib::worker::WorkerManager;
use vocal_manager_lib::AppState;

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            worker: WorkerManager::new(),
            model_sizes: Mutex::new(HashMap::new()),
        })
        .invoke_handler(tauri::generate_handler![
            vocal_manager_lib::commands::get_config,
            vocal_manager_lib::commands::save_config,
            vocal_manager_lib::commands::set_qwen_model,
            vocal_manager_lib::commands::set_backend,
            vocal_manager_lib::commands::check_status,
            vocal_manager_lib::commands::worker_start,
            vocal_manager_lib::commands::worker_speak,
            vocal_manager_lib::commands::worker_stop,
            vocal_manager_lib::commands::worker_state,
            vocal_manager_lib::commands::worker_logs,
            vocal_manager_lib::commands::mcp_status,
            vocal_manager_lib::commands::mcp_install_config,
            vocal_manager_lib::commands::mcp_test,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Vocal Manager");
}
