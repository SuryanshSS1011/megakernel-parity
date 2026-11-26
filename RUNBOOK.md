# GPU Runbook

This file lists every GPU run the study needs, in the order they should happen. Each step says what it would answer, what to run, roughly how long it takes and what to send back. Nothing here has been run on a GPU yet, so the first step is a go/no-go check, and later time estimates would be revised from its logs.

## 0. One-time setup on the A40 machine

```bash
git clone https://github.com/SuryanshSS1011/megakernel-parity && cd megakernel-parity
pip install -e ".[dev]"
python scripts/build_prompts.py                 # writes data/prompts/eval64.jsonl

git clone --recursive https://github.com/mirage-project/mirage ~/mirage
cd ~/mirage && git checkout ed771bda1b1143e660eb36d0f82e6e24e9df7b9c && git submodule update --init --recursive
pip install -e . -v                             # needs CMake >= 3.24 and nvcc on PATH
cd -
python -c "import mirage; print(mirage.get_configurations_from_gpu(0))"   # expect (64, ...) on an A40
python scripts/env_report.py --mirage-root ~/mirage > results/env_$(hostname).json
```

MPK's own CI only runs on B200 GPUs, so its Ampere path is not continuously tested. If `pip install -e .` or the import fails, send the error output and stop there.

## 1. Smoke test on the A40 (go/no-go)

**Question:** Does MPK build and run Qwen3 on sm_86, and does every arm work?

```bash
python experiments/mpk/run_mpk.py --mirage-root ~/mirage --model Qwen/Qwen3-1.7B --profile smoke
python experiments/mpk/run_mpk.py --mirage-root ~/mirage --model Qwen/Qwen3-0.6B --profile smoke
python scripts/analyze.py results/mpk
```

This would be 7 launches per model on one prompt with 64 new tokens. Each arm compiles its own megakernel once, so most of the time is nvcc. Expect roughly 30 to 90 minutes in total.

**Send back:** `results/mpk/` including the `.log` files, and `results/env_*.json`. The logs show compile time per arm, which sets the real cost of steps 3 and 4.

## 2. Kernel-level checks on the A40

**Question:** Which MPK tasks keep a row bit-identical when the row count or task count changes?

```bash
python experiments/mpk/kernel_parity.py --out results/mpk_kernels/a40.json
```

This would build about 11 small test-mode kernels covering linear layers at M = 1 to 8, the vocab projection at 64 and 96 tasks, and causal attention at 1 to 16 tokens. Expect 15 to 45 minutes. A check that errors is recorded and the rest continue, so send the JSON and the console output either way.

## 3. Full sweep on the A40

**Question:** Does MPK speculative decoding reproduce plain decoding? Is MPK run-to-run, batch and configuration invariant?

```bash
python experiments/mpk/run_mpk.py --mirage-root ~/mirage --model Qwen/Qwen3-1.7B --profile full
python scripts/analyze.py results/mpk
```

This would be 444 launches with 512 new tokens each. Kernels are compiled once per arm and reused after that, and the `ns_nocache` arm checks that reuse does not change results. The run is resumable, so it can be stopped and restarted with the same command. Expect several hours, which makes it a good overnight job.

**Send back:** `results/mpk/` and `results/summary.md`.

## 4. Cross-GPU run on the A100

**Question:** Does the same MPK build produce the same tokens on 108 SMs as on 84 SMs?

Repeat step 0 on the A100 machine with the same mirage commit, then run this.

```bash
python experiments/mpk/run_mpk.py --mirage-root ~/mirage --model Qwen/Qwen3-1.7B --profile cross
python experiments/mpk/kernel_parity.py --out results/mpk_kernels/a100.json
```

This would be 168 launches. Copy the A40's `results/mpk` next to the A100's results, then `python scripts/analyze.py results/mpk` adds the cross-GPU row automatically.

## 5. Hazy megakernel on a Modal H100

**Question:** Is the Hazy low-latency megakernel run-to-run deterministic?

You need a Modal account and a Hugging Face token that has been granted access to `meta-llama/Llama-3.2-1B-Instruct`. Request that access early, because approval can take a while.

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_xxx
modal run experiments/hazy/modal_app.py --limit 2 --reps 3 --ntok 64        # first, about 5 GPU-minutes
modal run experiments/hazy/modal_app.py --limit 32 --reps 20 --ntok 256     # then the full run
python scripts/analyze.py results/hazy
```

The image compiles the megakernel without a GPU, so a build failure costs no GPU time. The full run should take roughly 20 to 40 H100-minutes, or about $1.50 to $3 at $3.95 per hour.

**Send back:** `results/hazy/`, or the image build log if the build fails.

## 6. Margin analysis, after steps 3 to 5

**Question:** Is each divergence a near-tie in fp32, which would point to rounding, or a large gap, which would point to a logic bug?

```bash
python experiments/margins/margin_at_divergence.py --runs results/mpk/NVIDIA_A40/Qwen_Qwen3-1.7B \
    --a ns --b spec_k3 --model Qwen/Qwen3-1.7B --out results/margins/a40_ns_vs_k3.json
```

Run it once for each arm pair that `results/summary.md` shows diverging. It loads the model in fp32, so a 1.7B model needs about 7 GB of weights, which fits on the A40 or A100.

## Optional runs

1. **Hazy on the A100:** The Hazy Makefile has a `GPU=A100` option, but the low-latency kernel uses Hopper TMA instructions, so it is expected to fail to compile. A failed attempt takes about 15 minutes. A success would remove the need for Modal.
2. **Qwen3-8B on the A40:** The same full sweep with `--model Qwen/Qwen3-8B` would test whether the findings hold at a production model size. It would only be worth running after step 3 works.
3. **RTX 5060:** Nothing is planned yet. MPK rejects sm_120 today, and the port attempt is scheduled for week 3.
