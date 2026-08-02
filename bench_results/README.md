# TTS Backend Benchmark — 4-way comparison

Same short text fed to each backend via the stdin worker protocol (same path the
Rust host uses). Each backend announces its own name first, then speaks the same
3 test lines — so you can tell who is who by ear.

**Test lines (identical for all backends, each preceded by its own name):**
```
This is <backend name> speaking.
This is a test of the voice engine.
I hope you can hear me clearly.
Testing one two three.
```

## Results (one run, this Mac, MLX/Metal)

| backend           | wall (s) | peak RSS (MB) | avg RSS (MB) |
|-------------------|----------|---------------|--------------|
| swift_chatterbox  | 10.76    | 514           | 478          |
| py_chatterbox     | 18.12    | 1079          | 694          |
| swift_qwen_06     | 18.46    | 1722          | 651          |
| swift_qwen_17     | 14.37    | 2211          | 1160         |

Notes:
- **peak RSS** = `maximum resident set size` from `/usr/bin/time -l` (macOS).
- **avg RSS** = process RSS sampled every 0.25 s for the worker's lifetime.
- All 4 backends produced audible speech from the same text.

## How to re-run

```bash
scripts/bench_tts.sh   # writes bench_results/<name>/*, prints the summary table
```

Raw per-run artifacts: `stdout.log`, `time.txt`, `rss_samples.txt`, `input.txt`
under `bench_results/<backend>/`.
