# megakernel-parity

Characterizing and mitigating run-to-run, verify-vs-decode and cross-GPU numerical divergence in LLM megakernels.

Megakernels fuse a whole forward pass into one persistent GPU kernel. They are fast, but their reduction schedules can change results between runs, between plain decode and speculative-decoding verification, and between GPUs with different SM counts. This project would measure those effects in open-source megakernels, trace each mismatch to a kernel, and test fixes that restore bitwise parity at low cost.

## Targets

| System | Hardware | Question |
|---|---|---|
| Mirage Persistent Kernel (MPK) | A40 (sm_86), A100 (sm_80) | Does spec-decode output match plain decode, and does output match across SM counts? |
| Hazy Megakernels, low-latency Llama-1B | H100 on Modal | Is the output run-to-run deterministic given bf16 `store_add` partial sums? |
| Hugging Face Transformers, baseline | A40 | How far do stock kernels drift between verify and decode on the same models? |
| MPK port, stretch goal | RTX 5060 (sm_120) | Can the Ampere task path run on consumer Blackwell? |

## Layout

| Path | Contents |
|---|---|
| `src/mkparity/` | Bitwise helpers, the TinyLlama model and the reduction-schedule emulator |
| `harness/` | GPU parity harnesses for Hugging Face (`parity_hf.py`) and vLLM (`parity_vllm.py`) |
| `experiments/mpk/` | MPK sweep runner (`run_mpk.py`), its demo patch and the test-mode kernel checks |
| `experiments/hazy/` | Hazy run-to-run script and the Modal H100 wrapper |
| `experiments/margins/` | fp32 margin analysis at each divergence |
| `scripts/` | Prompt builder, environment report and the results summarizer |
| `prompts/eval64_ids.json` | The ids of the fixed 64-prompt evaluation set |
| `RUNBOOK.md` | Every GPU task the study needs, in order, with the steps only a person can do |
| `CLAUDE.md` | Rules for a coding agent running the RUNBOOK tasks on the cluster |
| `slurm/` | ROAR Collab setup, the `submit.sh` wrapper and one batch script per task |
| `env.example` | Template for the git-ignored `.env` that holds the token and Slurm account |
| `docs/run_log.md` | Log of every GPU job, its outcome and any fix it needed |
| `experiments/emulator/` | CPU characterization scripts and the roofline cost model |
| `results/emulator/` | Results from the first emulator runs (Nov 21, 2026) |
| `tests/` | pytest suite that runs on CPU |
| `docs/plan.md` | The project plan |
| `docs/evaluation.md` | The evaluation framework: comparisons, metrics, protocol and pass criteria |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

# reproduce the emulator study (CPU, about 25 minutes on 2 cores)
mkdir -p data
curl -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python experiments/emulator/train_char.py --steps 700
python experiments/emulator/run_characterization.py
python experiments/emulator/run_long.py
```

## Status

The CPU emulator results are directional only, because they model reduction schedules and not real kernels. No GPU measurements have been taken yet. See `RUNBOOK.md` for the runs that would produce them.
