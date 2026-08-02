#!/usr/bin/env bash
# Replaces the whole benchmark script: each backend runs through a tiny
# launcher that env-set + execs the worker, so /usr/bin/time (which execs the
# launcher) reports the worker's true RSS, and the sampled PID is the worker.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench_results"
mkdir -p "$OUT"

# Each backend announces its own name in the audio (first line), then speaks the
# same 3 test lines — so listening back you can tell who is who.
COMMON_LINES="This is a test of the voice engine.
I hope you can hear me clearly.
Testing one two three."

# per-backend input: name announcement + common lines
write_input() {
    local name="$1" dir="$2"
    printf 'This is %s speaking.\n%s\n' "$name" "$COMMON_LINES" > "$dir/input.txt"
}

CHATTERBOX_MODEL_PATH="/Users/singupallikartik/.cache/huggingface/hub/models--mlx-community--chatterbox-turbo-4bit/snapshots/c63817725071d7b5269c7b558772d6e8cbf59cec"
QWEN_06="/Users/singupallikartik/Developer/fun-projects/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
QWEN_17="/Users/singupallikartik/Developer/fun-projects/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
ENGINE_BIN="$ROOT/engine/.build/release"
PYTHON="$ROOT/src/.venv/bin/python"
CHATTERBOX_WORKER="$ROOT/scripts/chatterbox_worker.py"

# --- per-backend launcher scripts (export env + exec the worker directly so
#     /usr/bin/time's child IS the worker; cwd=engine dir) ---
LAUNCHERS="$OUT/launchers"
mkdir -p "$LAUNCHERS"
ENGINE_DIR="$ROOT/engine"

cat > "$LAUNCHERS/swift_chatterbox" <<EOF
#!/usr/bin/env bash
cd "$ENGINE_DIR"
export CHATTERBOX_MODEL_PATH="$CHATTERBOX_MODEL_PATH"
exec "$ENGINE_BIN/ChatterboxWorker"
EOF

cat > "$LAUNCHERS/py_chatterbox" <<EOF
#!/usr/bin/env bash
cd "$ROOT"
export CHATTERBOX_MODEL="mlx-community/chatterbox-turbo-4bit"
exec "$PYTHON" "$CHATTERBOX_WORKER"
EOF

cat > "$LAUNCHERS/swift_qwen_06" <<EOF
#!/usr/bin/env bash
cd "$ENGINE_DIR"
export VOCAL_MODEL_PATH="$QWEN_06"
export VOCAL_SPEAKER="Dylan"
export VOCAL_LANGUAGE="English"
export VOCAL_INSTRUCT="be very Fast, Serious, and not skip any word like you are reading audiobook"
export VOCAL_TEMPERATURE="0.8"
exec "$ENGINE_BIN/VocalWorker"
EOF

cat > "$LAUNCHERS/swift_qwen_17" <<EOF
#!/usr/bin/env bash
cd "$ENGINE_DIR"
export VOCAL_MODEL_PATH="$QWEN_17"
export VOCAL_SPEAKER="Dylan"
export VOCAL_LANGUAGE="English"
export VOCAL_INSTRUCT="be very Fast, Serious, and not skip any word like you are reading audiobook"
export VOCAL_TEMPERATURE="0.8"
exec "$ENGINE_BIN/VocalWorker"
EOF
chmod +x "$LAUNCHERS"/*

sample_rss() {
    local pid="$1" outfile="$2"
    : > "$outfile"
    while kill -0 "$pid" 2>/dev/null; do
        local rss_kb
        rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        if [[ -n "$rss_kb" ]]; then echo "$rss_kb" >> "$outfile"; fi
        sleep 0.25
    done
}

run_case() {
    local name="$1" launcher="$2" announce="$3"
    local dir="$OUT/$name"
    mkdir -p "$dir"
    echo "=== $name ==="
    write_input "$announce" "$dir"

    local t_out="$dir/time.txt" sample_out="$dir/rss_samples.txt"
    /usr/bin/time -l "$launcher" < "$dir/input.txt" >"$dir/stdout.log" 2>"$t_out" &
    local tpid=$!

    # /usr/bin/time forks the launcher (which execs the worker). Find the worker
    # child PID and sample IT for average RSS.
    local worker_pid=""
    for _ in $(seq 1 20); do
        worker_pid=$(pgrep -P "$tpid" 2>/dev/null | head -1)
        [[ -n "$worker_pid" ]] && break
        sleep 0.1
    done
    if [[ -z "$worker_pid" ]]; then worker_pid="$tpid"; fi
    sample_rss "$worker_pid" "$sample_out" &
    local s_pid=$!
    wait "$tpid"
    local rc=$?
    wait "$s_pid" 2>/dev/null

    local peak_bytes=0 avg_bytes=0 n=0 sum=0
    local pk
    pk=$(awk '/maximum resident set size/{print $1}' "$t_out")
    peak_bytes=${pk:-0}
    if [[ -s "$sample_out" ]]; then
        n=$(wc -l < "$sample_out" | tr -d ' ')
        sum=$(awk '{s+=$1} END{print s}' "$sample_out")
        avg_bytes=$(( (sum * 1024) / n ))
    fi
    local wall
    wall=$(awk '/real/{print $1}' "$t_out")
    echo "  rc=$rc wall=${wall}s peak=$((peak_bytes/1048576))MB avg=$((avg_bytes/1048576))MB samples=$n"
    echo -e "$name\t$wall\t$peak_bytes\t$avg_bytes\t$n" >> "$OUT/summary.tsv"
}

rm -f "$OUT/summary.tsv"
echo -e "backend\twall_s\tpeak_bytes\tavg_bytes\trss_samples" > "$OUT/summary.tsv"

echo "🎧 NOW PLAYING: Swift Chatterbox (announces itself)"
run_case "swift_chatterbox" "$LAUNCHERS/swift_chatterbox" "Swift Chatterbox"
echo "🎧 NOW PLAYING: Python Chatterbox (announces itself)"
run_case "py_chatterbox"     "$LAUNCHERS/py_chatterbox" "Python Chatterbox"
echo "🎧 NOW PLAYING: Swift Qwen 0.6B (announces itself)"
run_case "swift_qwen_06"     "$LAUNCHERS/swift_qwen_06" "Swift Qwen 0.6B"
echo "🎧 NOW PLAYING: Swift Qwen 1.7B (announces itself)"
run_case "swift_qwen_17"     "$LAUNCHERS/swift_qwen_17" "Swift Qwen 1.7B"

echo
echo "=== SUMMARY ==="
column -t -s $'\t' "$OUT/summary.tsv"
