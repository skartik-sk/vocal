#!/usr/bin/env bash
# 4-way TTS A/B: native Swift Chatterbox (multilingual Hindi) vs Python
# Chatterbox (multilingual Hindi) vs Swift Qwen 0.6B vs Swift Qwen 1.7B.
#
# Each backend announces its model name in English, then speaks a mixed
# Hindi+English script (code-switching). Peak RAM is captured with
# /usr/bin/time -l (maximum resident set size). Swift binaries must run
# with cwd = engine/ so MLX finds the metallib.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench_results/hindi_ab"
mkdir -p "$OUT"

ENGINE_BIN="$ROOT/engine/.build/release"
ENGINE_DIR="$ROOT/engine"
PYTHON="$ROOT/src/.venv/bin/python"
QWEN_06="$HOME/Developer/fun-projects/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
QWEN_17="$HOME/Developer/fun-projects/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"

SWIFT_CB_TEXT="This is Swift Chatterbox speaking. नमस्ते दोस्तों, this is the real test. मैं हिंदी और English दोनों बोल सकता हूँ। यह बहुत अच्छा है, right? क्या आप मुझे सुन सकते हैं? one two three."
PY_CB_TEXT="$SWIFT_CB_TEXT"
QWEN_TEXT="This is Swift Qwen speaking. Hello friends, this is the real test. I can speak Hindi and English both. It is very good, right? Can you hear me? One two three."

peak_gb() { # file -> GB
    local f="$1"
    local b
    b=$(grep -oE "[0-9]+  maximum resident set size" "$f" | awk '{print $1}')
    if [ -n "$b" ]; then
        echo "$b" | awk '{printf "%.2f GB", $1/1024/1024/1024}'
    else
        echo "?"
    fi
}

echo "== 4-way A/B (English announce + Hindi/English mix) =="

echo "[1/4] Swift Chatterbox (multilingual Hindi)..."
(cd "$ENGINE_DIR" && CHATTERBOX_ML_MODEL="/tmp/chatterbox-4bit" CHATTERBOX_ML_OUT="$OUT/swift_cb.wav" \
  /usr/bin/time -l "$ENGINE_BIN/ChatterboxMLWorker" "$SWIFT_CB_TEXT" 2> "$OUT/swift_cb.time") | sed 's/^/   /'
echo "   peak: $(peak_gb "$OUT/swift_cb.time")"

echo "[2/4] Python Chatterbox (multilingual Hindi)..."
cat > "$OUT/py_cb_gen.py" <<'PYEOF'
import os, sys
import mlx.core as mx, numpy as np, soundfile as sf
from mlx_audio.tts.utils import load_model, get_model_path
from mlx_audio.tts.models.chatterbox.tokenizer import MTLTokenizer
m = load_model(get_model_path("/tmp/chatterbox-4bit"))
tok = MTLTokenizer("/tmp/chatterbox-4bit/tokenizer.json")
text = os.environ["PY_TEXT"]
tt = tok.text_to_tokens(text, language_id="hi")
toks = m.t3.inference(t3_cond=m._conds.t3, text_tokens=tt, max_new_tokens=150, cfg_weight=0.5)
mx.eval(toks)
wav, _ = m.s3gen.inference(toks, m._conds.gen)
mx.eval(wav)
sf.write(os.environ["PY_OUT"], np.array(wav[0]), 24000)
print("  py tokens:", toks.shape[1], "wav:", len(np.array(wav[0])), file=sys.stderr)
PYEOF
(cd "$ROOT" && PY_TEXT="$PY_CB_TEXT" PY_OUT="$OUT/py_cb.wav" \
  /usr/bin/time -l "$PYTHON" "$OUT/py_cb_gen.py" 2> "$OUT/py_cb.time") | sed 's/^/   /'
echo "   peak: $(peak_gb "$OUT/py_cb.time")"

echo "[3/4] Swift Qwen 0.6B (English announce + English translation)..."
(cd "$ENGINE_DIR" && /usr/bin/time -l "$ENGINE_BIN/Qwen3TTSDemo" --text "$QWEN_TEXT" \
  --model "$QWEN_06" --output "$OUT/qwen_06.wav" --language en --speaker Dylan 2> "$OUT/qwen_06.time") | sed 's/^/   /'
echo "   peak: $(peak_gb "$OUT/qwen_06.time")"

echo "[4/4] Swift Qwen 1.7B (English announce + English translation)..."
(cd "$ENGINE_DIR" && /usr/bin/time -l "$ENGINE_BIN/Qwen3TTSDemo" --text "$QWEN_TEXT" \
  --model "$QWEN_17" --output "$OUT/qwen_17.wav" --language en --speaker Dylan 2> "$OUT/qwen_17.time") | sed 's/^/   /'
echo "   peak: $(peak_gb "$OUT/qwen_17.time")"

echo ""
echo "=== Peak RAM (maximum resident set size) ==="
for f in swift_cb py_cb qwen_06 qwen_17; do
    [ -f "$OUT/$f.wav" ] && echo "  $f.wav: $(stat -f%z "$OUT/$f.wav") bytes | peak $(peak_gb "$OUT/$f.time")"
done
echo "Wavs in $OUT/"
