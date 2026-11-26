# Emulator results

These files come from the first CPU runs on November 21, 2026. They used a 3.8M-parameter char-level TinyLlama trained for 700 steps on TinyShakespeare, which reached a validation loss of 1.62.

| File | Produced by | Setting |
|---|---|---|
| `characterization.json` | `experiments/emulator/run_characterization.py` | 16 prompts, 128 tokens, 5 seeds |
| `long_runs.json` | `experiments/emulator/run_long.py` | 40 prompts, 384 tokens |
| `cost_model.json` | `experiments/emulator/cost_model.py` | Roofline estimates for H100, A100 40GB and A40 |

The emulator models reduction schedules, not real GPU kernels, so these numbers are directional only. Schedule IDs are defined in `src/mkparity/schedules.py`.
