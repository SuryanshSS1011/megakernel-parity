# GPU Runbook

This file lists every GPU task the study needs, in order, for a coding agent such as Claude Code. Part 1 runs on ROAR Collab's A40 and A100 GPUs through Slurm. Part 2 runs the Hazy megakernel on a Modal H100, because ROAR has no H100. No GPU measurement has been taken yet, so the time estimates are first guesses that T4's logs would revise.

The agent rules live in `CLAUDE.md`. They cover secrets, stop conditions and what to commit. Settings and the HF token come from `.env` in the repository root, which git ignores.

## Overview

| Task | Job | GPU | Depends on | Estimate |
|---|---|---|---|---|
| T1 | `probe` | A40 | none | 15 min |
| T2 | `slurm/setup.sh` on the login node | none | T1 | 30 to 60 min |
| T3 | `build` | A40 | T2 | 30 to 90 min |
| T4 | `smoke` | A40 | T3 | 1 to 3 h |
| T5 | `kernels` | A40 | T4 | 0.5 to 1 h |
| T6 | `hf_baselines` | A40 | T2 | 1 to 2 h |
| T7 | `full` | A40 | T4 | 8 to 14 h |
| T8 | `cross` | A100 | T4 | 4 to 7 h |
| T9 | `margins` | A40 | T7, T8 | 1 to 2 h |
| T10 | `modal run` from the login node | Modal H100 | T2 | 20 to 40 H100-min |

T5, T6, T7 and T8 can run at the same time once T4 passes, because they use different GPUs or write to different folders. Part 1 would take roughly 17 to 30 GPU-hours on ROAR. T10 only needs T2, so it could start early, but it is listed last because it is the only task that costs money.

## Part 1: ROAR Collab (A40 and A100)

Every GPU task here runs as `slurm/submit.sh <job> --gpu a40|a100`, which reads the account from `.env` and prints the job id. Jobs write their console output to `slurm/logs/`.

### T1. Check the settings and probe a GPU node

First confirm that `.env` exists and sets `HF_TOKEN`, without printing it.

```bash
set -a; source .env; set +a
[[ -n "$HF_TOKEN" ]] && echo "HF_TOKEN set" || echo "HF_TOKEN missing"
[[ -n "$ROAR_ACCOUNT" ]] && echo "account $ROAR_ACCOUNT" || sacctmgr show assoc user=$USER format=account%30,partition%20
```

If `ROAR_ACCOUNT` is missing and `sacctmgr` lists exactly one account, append `ROAR_ACCOUNT=<name>` and `ROAR_PARTITION=standard` to `.env`, or `sla-prio` if the account is an allocation. If it lists several, ask the user which one to charge. Then submit the probe.

```bash
slurm/submit.sh probe --gpu a40
```

**Purpose:** The probe records the driver version, the CUDA and Python modules, partition time limits, and whether compute nodes reach the internet.

**Then:** Pick the newest CUDA 12.x module whose version does not exceed the CUDA version `nvidia-smi` reports, and a Python 3.10 to 3.12 module. Append them to `.env` as `MKP_CUDA_MODULE` and `MKP_PYTHON_MODULE`. Jobs read models from the cache whether or not compute nodes reach the internet, so nothing else changes.

### T2. Set up on the login node

```bash
bash slurm/setup.sh
```

**Purpose:** Setup creates the virtual environment in work storage, installs a torch build that matches the CUDA module, clones mirage at `ed771bd` and Hazy at `7309cec`, prefetches the Rust crates, downloads Qwen3-1.7B, Qwen3-0.6B and Llama-3.2-1B-Instruct into scratch and builds the prompt set. Each step can be rerun alone, for example `bash slurm/setup.sh models`.

**Acceptance:** `data/prompts/eval64.jsonl` has 64 lines, and the model step prints `ok` for all three models. A Llama failure here means the token lacks gated-repo read access, which only the user can fix, so report it and continue with the Qwen tasks.

### T3. Build MPK

```bash
B=$(slurm/submit.sh build --gpu a40)
```

**Purpose:** One build serves both GPUs, because mirage compiles its runtime for sm_75 through sm_90.

**Acceptance:** The log ends with `build ok` and prints the worker and scheduler counts, and `results/env/A40.json` exists. MPK's own CI only runs on B200 GPUs, so its Ampere path is not continuously tested. A failure here is a real finding, and CLAUDE.md says when to stop.

### T4. Smoke test, the go/no-go gate

```bash
slurm/submit.sh smoke --gpu a40 --after "$B"
```

**Purpose:** The smoke test sends one prompt through all 7 arms for Qwen3-1.7B and Qwen3-0.6B, with 64 new tokens each. It answers whether MPK runs on sm_86 and whether every arm works.

**Acceptance:** All 14 launches return 0, and `results/summary_smoke_A40.md` lists every arm. The logs show the compile time per arm, so revise the T7 and T8 estimates from them and record the new figures in `docs/run_log.md`. Open the `roar-runs` pull request after this task.

### T5. Kernel checks on the A40

```bash
slurm/submit.sh kernels --gpu a40
```

**Purpose:** These checks test which MPK tasks keep a row bit-identical when the row count, task count or token count changes. They cover linear layers at M = 1 to 8, the vocab projection at 64 and 96 tasks, and causal attention at 1 to 16 tokens.

**Acceptance:** `results/mpk_kernels/A40.json` exists. A check that errors is recorded and the rest continue, so a partial file still counts.

### T6. Hugging Face baselines

```bash
slurm/submit.sh hf_baselines --gpu a40
```

**Purpose:** This task measures verify(k+1) against decode(1) on stock Hugging Face kernels for Llama-3.2-1B-Instruct and Qwen3-1.7B in three settings, bf16 SDPA, bf16 eager and fp32 SDPA. It gives the non-megakernel baseline, and its Llama rows link the MPK results to the Hazy results, since Hazy runs that model. It only needs T2, so it can run while T3 and T4 are going.

**Acceptance:** Each of the six `results/hf_parity/A40/*.jsonl` files exists and its log ends without a traceback.

### T7. Full sweep on the A40

```bash
F=$(slurm/submit.sh full --gpu a40)
```

**Purpose:** The sweep answers whether MPK speculative decoding reproduces plain decoding, and whether MPK is run-to-run, batch and configuration invariant. It runs 444 launches with 512 new tokens each. Kernels compile once per arm and are reused, and the `ns_nocache` arm checks that reuse does not change results.

**Acceptance:** `results/summary.md` has a row for every arm against `ns`, and fewer than 2% of launches failed. The job is resumable, so if it hits the time limit, submit it again and it would pick up where it stopped. If T1 showed a partition limit under 24 hours, submit with `-- --time=<limit>` and resubmit as often as needed.

### T8. Cross-GPU run on the A100

```bash
X=$(slurm/submit.sh cross --gpu a100)
```

**Purpose:** This run tests whether identical builds and prompts give identical tokens on 108 SMs and 84 SMs. It runs 168 launches and then the kernel checks on sm_80. The results land next to the A40 results on the shared filesystem, so `analyze.py` adds the cross-GPU row without any copying.

**Acceptance:** `results/summary.md` shows an A40 against A100 row, and `results/mpk_kernels/A100-*.json` exists.

### T9. Margin analysis

Run this after T7 and T8. Pass the arm pairs that `results/summary.md` shows diverging.

```bash
slurm/submit.sh margins --gpu a40 --env MKP_PAIRS="ns:spec_k3 ns:spec_k5"
```

**Purpose:** The analysis replays each shared prefix through an fp32 model and records the top-2 margin at the first divergence. A near-tie in fp32 would point to rounding, and a large gap would point to a logic bug.

**Acceptance:** There is one `results/margins/*.json` per GPU and pair. Then write a short findings note in `docs/run_log.md` covering exact-match rates with their 95% intervals, median first-divergence positions and margin distributions.

### Optional ROAR runs, only when the user asks

1. **O1, Hazy on the A100:** `slurm/submit.sh hazy_a100_build --gpu a100` tries the Hazy build with `GPU=A100`. The low-latency kernel uses Hopper TMA instructions, so it is expected to fail, and the log is the result either way. A success would move T10 off Modal.
2. **O2, MPK on Llama-3.2-1B-Instruct:** Both megakernels would then run the same model. MPK's `demo/llama3/demo.py` lacks the `--prompt`, `--save-tokens`, `--ignore-eos` and `--max-new-tokens` flags the runner relies on, so this needs a second patch set in `patch_demo.py` and a `--demo llama3` option in `run_mpk.py`. Two risks are known. Llama-3.2-1B uses a head dimension of 64, while `tasks/ampere/single_batch_gqa.cuh` hard-codes 128 and `rotary_embedding.cuh` defaults to 128. The model also ties its input and output embeddings. The port would pass if the `ns` arm matches the `torch` arm on the first 16 tokens of the smoke prompt, and it would be capped at 6 GPU jobs before stopping with a written explanation.
3. **O3, Qwen3-8B on the A40:** `slurm/submit.sh full --gpu a40 --env MKP_MODEL=Qwen/Qwen3-8B` would test whether the findings hold at a production size. It needs `MKP_MODELS=Qwen/Qwen3-8B bash slurm/setup.sh models` first, and it fits in 48 GB.
4. **O4, vLLM baseline:** `harness/parity_vllm.py` would need its own virtual environment, because vLLM pins its own torch. It would compare the `ref` arm with the `ngram` arm at k = 4 on the A100.

## Part 2: H100 on Modal

### T10. Hazy run-to-run determinism

This task runs from the ROAR login node, because Modal does the GPU work remotely. It needs the Modal login in `~/.modal.toml` and the environment from T2. If `~/.modal.toml` does not exist, ask the user to run `modal token new` and continue with Part 1 meanwhile.

```bash
source "${MKP_WORK:-/storage/work/$USER}/envs/mkparity/bin/activate"
bash scripts/modal_hf_secret.sh                                              # copies HF_TOKEN from .env into Modal safely
modal run experiments/hazy/modal_app.py --limit 2 --reps 3 --ntok 64         # check run, about 5 GPU-minutes
modal run experiments/hazy/modal_app.py --limit 32 --reps 20 --ntok 256      # full run
python scripts/analyze.py results/mpk results/hazy --out results/summary
```

**Purpose:** The run tests whether the Hazy low-latency megakernel gives identical logits across repeated runs, given its bf16 `store_add` partial sums. The image compiles the megakernel without a GPU, so a build failure costs no GPU time.

**Acceptance:** `results/hazy/<gpu>/e3_l32_r20_n256/` holds one file per prompt, and the summary shows per-step hash agreement across repetitions. The full run would take roughly 20 to 40 H100-minutes, about $1.50 to $3 at $3.95 per hour. If the image build fails, commit its log and stop T10 there. Once the verdict is in, the M1 fixed-order combine from `docs/plan.md` would be the next H100 run, but only when the user asks for it.
