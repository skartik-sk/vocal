#!/usr/bin/env python3
"""Chatterbox-Turbo TTS worker — drop-in backend for Vocal.

Speaks the SAME stdin protocol as the Swift VocalWorker: read one sentence per
line, speak it, loop until stdin closes (EOF). Spawned by the Rust host (right-
click "Speak with Vocal") / vocal-mcp when the active backend is chatterbox.

Config via env:
  CHATTERBOX_MODEL     HF repo id or local path (default mlx-community/chatterbox-turbo-4bit)
  CHATTERBOX_REF_AUDIO optional wav path for voice cloning (default: built-in voice)

Run with the project venv (has mlx-audio 0.4.6):
  src/.venv/bin/python scripts/chatterbox_worker.py

Note: chatterbox-TURBO ignores emotion/exaggeration tuning (per mlx-audio), so
it reads text plainly; no emotion tags are injected. For a specific plain voice,
point CHATTERBOX_REF_AUDIO at a clean reading sample.
"""

import os
import sys
import tempfile

MODEL_ID = os.environ.get("CHATTERBOX_MODEL", "mlx-community/chatterbox-turbo-4bit")
REF_AUDIO = os.environ.get("CHATTERBOX_REF_AUDIO") or None
# mlx-audio writes a wav per call; send them to /tmp so the project dir stays clean.
_OUT_DIR = tempfile.gettempdir()

print(f"[chatterbox] loading {MODEL_ID} ...", flush=True)
from mlx_audio.tts.utils import load_model, get_model_path
from mlx_audio.tts import generate

_model = load_model(get_model_path(MODEL_ID))
print(
    f"[chatterbox] ready (voice={'clone:' + REF_AUDIO if REF_AUDIO else 'default'}). "
    "Waiting for sentences...",
    flush=True,
)

for line in sys.stdin:
    text = line.strip()
    if not text:
        continue
    try:
        generate.generate_audio(
            text=text,
            model=_model,
            ref_audio=REF_AUDIO,
            play=True,
            save=False,
            output_path=_OUT_DIR,
            file_prefix="vocal_chatterbox",
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[chatterbox] error: {e}", file=sys.stderr, flush=True)

print("[chatterbox] stdin closed. Exiting.", flush=True)
