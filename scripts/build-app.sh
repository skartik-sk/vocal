#!/usr/bin/env bash
#
# Build a SELF-CONTAINED Vocal.app.
#
# Bundles the native TTS worker (ChatterboxMLWorker), the Metal library
# (default.metallib), the chatterbox-4bit model (into Contents/Resources/), and
# the MCP server (vocal_mcp, into Contents/MacOS/) into one self-contained
# Vocal.app. The host AND the MCP server resolve all assets relative to the
# bundle — so the app (right-click Service + agent `speak` tool) runs with no
# repo, no build cache, and no HF cache on disk.
#
# Prereqs: Xcode CLI tools, Rust, `cargo install cargo-bundle`, and a downloaded
# chatterbox-4bit snapshot dir (passed as the first arg; see usage below).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- the model dir is the one asset we can't rebuild from source ---
# Pass the chatterbox-4bit snapshot dir as the first arg (or MODEL_DIR env).
# This is a BUILD-TIME source only — it's independent of vocal.config's runtime
# `chatterbox_model_path` (which points at the installed bundle, not the source).
MODEL_DIR="${1:-${MODEL_DIR:-}}"
if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
    echo "✗ Pass the chatterbox-4bit model dir as the first arg, e.g.:" >&2
    echo "    ./scripts/build-app.sh ~/.cache/huggingface/hub/models--mlx-community--chatterbox-4bit/snapshots/<hash>" >&2
    echo "  (download first with: huggingface-cli download mlx-community/chatterbox-4bit)" >&2
    exit 1
fi
for f in model.safetensors tokenizer.json conds.safetensors; do
    [[ -e "${MODEL_DIR}/${f}" ]] || { echo "✗ ${MODEL_DIR}/${f} not found" >&2; exit 1; }
done

echo "==> Building Swift engine (release)…"
swift build -c release --package-path engine

echo "==> Building Rust host + MCP (release)…"
cargo build --release

echo "==> Bundling Vocal.app (cargo bundle --release)…"
cargo bundle --release

APP="target/release/bundle/osx/Vocal.app"
RES="${APP}/Contents/Resources"
[[ -d "${RES}" ]] || { echo "✗ ${RES} missing — did cargo bundle succeed?" >&2; exit 1; }

echo "==> Copying worker + Metal lib into Contents/Resources/…"
cp "engine/.build/release/ChatterboxMLWorker" "${RES}/ChatterboxMLWorker"
chmod +x "${RES}/ChatterboxMLWorker"
cp "engine/default.metallib" "${RES}/default.metallib"

echo "==> Copying the MCP server into Contents/MacOS/…"
# Lives in MacOS/ (not Resources/) so is_bundled() detects the bundle and the
# server auto-resolves worker/metallib/model from Contents/Resources/ — letting
# the agent `speak` tool run with no repo / build cache / HF cache on disk.
cp "target/release/vocal_mcp" "${APP}/Contents/MacOS/vocal_mcp"
chmod +x "${APP}/Contents/MacOS/vocal_mcp"

echo "==> Copying model (dereferencing HF-cache symlinks) into Contents/Resources/chatterbox-4bit/…"
mkdir -p "${RES}/chatterbox-4bit"
for f in model.safetensors tokenizer.json conds.safetensors; do
    cp -L "${MODEL_DIR}/${f}" "${RES}/chatterbox-4bit/"
done

echo "==> Ad-hoc codesign (lets macOS Services + the nested worker run locally)…"
codesign --force --deep --sign - "${APP}"

echo
echo "✅ Built ${APP}"
du -sh "${APP}"
echo
echo "Next: drag Vocal.app into /Applications, then refresh macOS Services:"
echo "    /System/Library/CoreServices/pbs -flush     (or log out & back in)"
