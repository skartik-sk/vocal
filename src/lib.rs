//! Vocal shared library.
//!
//! Code shared between the macOS Services host (`src/main.rs`) and the MCP
//! server (`src/bin/vocal_mcp.rs`). Currently exposes the runtime config
//! parser that both binaries use to locate the engine binary, the model, and
//! voice defaults.

pub mod config;

pub use config::VocalConfig;
