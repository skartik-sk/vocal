#!/usr/bin/env python3
"""Speak text with Chatterbox-Turbo (4-bit) via mlx-audio.

Chatterbox is a different model family from Qwen3-TTS, so it runs through the
Python mlx-audio library (not the Swift engine). Use the project venv which has
mlx-audio 0.4.6 (with chatterbox + chatterbox_turbo support):

    src/.venv/bin/python scripts/chatterbox_speak.py "Hello, this is a test."

or pipe text in:

    echo "Hello" | src/.venv/bin/python scripts/chatterbox_speak.py

Optional:
  * Voice cloning: set CHATTERBOX_REF_AUDIO=/path/to/voice.wav
  * Emotion tags in the text: [laugh] [sigh] [chuckle] [groan] etc.
"""

import os
import subprocess
import sys

MODEL = "mlx-community/chatterbox-turbo-4bit"


def main() -> None:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        print('usage: chatterbox_speak.py "some text to speak"', file=sys.stderr)
        sys.exit(2)

    cmd = [
        sys.executable, "-m", "mlx_audio.tts.generate",
        "--model", MODEL,
        "--text", text,
        "--play",
    ]
    ref = os.environ.get("CHATTERBOX_REF_AUDIO")
    if ref:
        cmd += ["--ref_audio", ref]

    print(f"[chatterbox] model={MODEL} chars={len(text)} ref_audio={ref or 'none'}",
          flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
