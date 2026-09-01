#!/usr/bin/env bash
#
# Build a SELF-CONTAINED Vocal.app.
#
# Bundles the native TTS worker (ChatterboxMLWorker), the Metal library
# (default.metallib), the chatterbox-4bit model (into Contents/Resources/), the
# MCP server (vocal_mcp, into Contents/MacOS/), and the Vocal Manager GUI
# (Tauri, into Contents/Managers/) into one self-contained Vocal.app. The host,
# the MCP server, and the Manager resolve all assets relative to the bundle —
# so the app (right-click Services + agent `speak` tool + double-click GUI)
# runs with no repo, no build cache, and no HF cache on disk.
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
# Tip: when rebuilding, pass the CURRENTLY INSTALLED bundle's copy:
#   ./scripts/build-app.sh /Applications/Vocal.app/Contents/Resources/chatterbox-4bit
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

# --- Swift worker: rebuild only when the build cache is gone AND no installed
# app can donate its worker (keeps rebuilds cheap in both time and disk) ---
WORKER_BIN="engine/.build/release/ChatterboxMLWorker"
if [[ -f "${WORKER_BIN}" ]]; then
    echo "==> Swift engine already built — reusing ${WORKER_BIN}"
elif [[ -f "/Applications/Vocal.app/Contents/Resources/ChatterboxMLWorker" ]]; then
    echo "==> No Swift build cache — reusing worker from the installed Vocal.app"
    WORKER_BIN="/Applications/Vocal.app/Contents/Resources/ChatterboxMLWorker"
else
    echo "==> Building Swift engine (release)…"
    swift build -c release --package-path engine
fi
METALLIB="engine/default.metallib"
[[ -f "${METALLIB}" ]] || METALLIB="/Applications/Vocal.app/Contents/Resources/default.metallib"

echo "==> Building Rust host + MCP (release)…"
cargo build --release

echo "==> Building Vocal Manager GUI (Tauri, release)…"
(cd src-tauri && cargo build --release)
# Tauri v2 embeds the frontend into the binary at compile time, so the .app
# around it is a thin shell we assemble ourselves below — no tauri-cli needed.
# Stage the binary and reclaim the ~1.5G Tauri build cache RIGHT AWAY (before
# the model copy needs the disk space — it's fully reproducible anyway).
mkdir -p target/release
cp "src-tauri/target/release/vocal-manager" "target/release/vocal-manager.staged"
rm -rf src-tauri/target
echo "    (freed src-tauri/target)"
MANAGER_BIN="target/release/vocal-manager.staged"

echo "==> Bundling Vocal.app (cargo bundle --release)…"
cargo bundle --release

APP="target/release/bundle/osx/Vocal.app"
RES="${APP}/Contents/Resources"
[[ -d "${RES}" ]] || { echo "✗ ${RES} missing — did cargo bundle succeed?" >&2; exit 1; }

echo "==> Copying worker + Metal lib into Contents/Resources/…"
cp "${WORKER_BIN}" "${RES}/ChatterboxMLWorker"
chmod +x "${RES}/ChatterboxMLWorker"
cp "${METALLIB}" "${RES}/default.metallib"

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

echo "==> Embedding the Vocal Manager GUI into Contents/Managers/…"
[[ -f "${MANAGER_BIN}" ]] || { echo "✗ ${MANAGER_BIN} missing — did the Tauri build succeed?" >&2; exit 1; }
MANAGER_APP="${APP}/Contents/Managers/Vocal Manager.app"
mkdir -p "${MANAGER_APP}/Contents/MacOS" "${MANAGER_APP}/Contents/Resources"
cp "${MANAGER_BIN}" "${MANAGER_APP}/Contents/MacOS/vocal-manager"
chmod +x "${MANAGER_APP}/Contents/MacOS/vocal-manager"
cp "src-tauri/icons/icon.icns" "${MANAGER_APP}/Contents/Resources/icon.icns"
cat > "${MANAGER_APP}/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>vocal-manager</string>
    <key>CFBundleIdentifier</key><string>com.kartik.vocal.manager</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>Vocal Manager</string>
    <key>CFBundleDisplayName</key><string>Vocal Manager</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
    <key>CFBundleIconFile</key><string>icon.icns</string>
    <key>LSMinimumSystemVersion</key><string>10.15</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

echo "==> Ad-hoc codesign (lets macOS Services + the nested worker run locally)…"
codesign --force --deep --sign - "${MANAGER_APP}"
codesign --force --deep --sign - "${APP}"

echo
echo "✅ Built ${APP}"
du -sh "${APP}"
echo
echo "Next: drag Vocal.app into /Applications, then refresh macOS Services:"
echo "    /System/Library/CoreServices/pbs -flush     (or log out & back in)"
