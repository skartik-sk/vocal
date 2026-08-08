//! Vocal shared library.
//!
//! Code shared between the macOS Services host (`src/main.rs`) and the MCP
//! server (`src/bin/vocal_mcp.rs`). Currently exposes the runtime config
//! parser that both binaries use to locate the engine binary, the model, and
//! voice defaults.

pub mod config;
pub mod worker;

pub use config::{VocalConfig, VOCAL_ROOT};

/// Split a paragraph into sentences on `.`, `?`, `!`. Shared by the macOS
/// Services host and the MCP server so long text is streamed to the worker one
/// sentence per line (each generation is capped by CHATTERBOX_ML_MAX_TOKENS).
pub fn split_paragraph(text: &str) -> Vec<String> {
    text.split_terminator(&['.', '?', '!'][..])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_punctuation_and_drops_empty() {
        let out = split_paragraph("Hello world. How are you? Fine!  ..");
        assert_eq!(out, vec!["Hello world", "How are you", "Fine"]);
    }
}
