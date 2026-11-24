# Characterizing and Mitigating Batch Invariance in LLM Megakernels

Project plan, November 21, 2026. Hardware updated November 24, 2026.

## 1. Bottom line

The megakernel direction is viable, and it is now better grounded than the vLLM-only version. Reading the megakernel source code shows two concrete, different numerical hazards. The Hazy low-latency megakernel accumulates down-projection partial sums into a bf16 residual with TMA `store_add`, so its result can depend on which SM finishes first. Mirage Persistent Kernel (MPK) runs speculative-decoding verification through a different attention kernel than plain decode, so verify and decode use different arithmetic. Production megakernels from Cohere and LMSYS report speed and benchmark accuracy only, and none of them reports bitwise run-to-run or verify-versus-decode parity.

CPU emulation of these reduction schedules on a small trained model reproduces both hazards. It also shows that two cheap schedule changes remove them completely in the emulator. The project would test whether the same holds on real megakernels.

The most important practical finding is that MPK has an Ampere backend. The core experiments could therefore run on the team's A40 and A100 GPUs at no cost, with Modal H100 time reserved for the Hazy megakernel.

## 2. Verification of earlier work

| Earlier claim | Status on Nov 21, 2026 |
|---|---|
| vLLM spec-decode tests tolerate mismatch (60% EAGLE, 66% speculators, 80% MTP, 90% LoRA+EAGLE3 under `VLLM_BATCH_INVARIANT=1`) | Still true at vLLM main commit 80456e5 |
| vLLM PR #63275 shows bitwise spec-vs-no-spec parity under batch-invariant mode | Verified. It was merged October 29, 2026. It reports 100% bitwise raw logprobs for Qwen3-8B with EAGLE3 and Qwen3-0.6B with ngram on GB300, and it documents a top-k/top-p gap |
| vLLM PR #62193 adds an env var `VLLM_BI_DRAFTER` | Corrected. The PR uses a `batch_invariance_exempt()` context manager, not an env var. It is still open, and it reports 36/36 bitwise matches on GH200 |
| llama.cpp FA dispatch picks VEC at one query row and MMA otherwise | Still true on master |
| Modal H100 costs about $3.95 per hour, and Starter includes $30 per month | Consistent across sources checked November 16 |
| The vLLM hypothesis H1 ("batch-invariant mode closes verify-vs-decode") is open | Largely closed on datacenter Blackwell by #63275. This is another reason to move to megakernels |

## 3. What the megakernel code shows

### 3.1 Hazy Megakernels (low-latency Llama-1B demo)

1. The main branch was last updated June 2, 2025, and it builds only for H100 or B200.
2. In `demos/low-latency-llama/matvec_adds.cu`, down-projection and o-projection results are written with `kittens::tma::store_add_async` from a bf16 shared-memory vector into the bf16 `hidden_states` residual.
3. The scheduler splits the 8192-wide down-projection reduction into 4 blocks of 2048. Four partial sums are therefore added into each residual element, and each add rounds to bf16. With three or more operands the sum depends on arrival order, which depends on SM timing.
4. The o-projection uses a single reduction block, so each element receives one add. A two-operand floating-point add is commutative, so that step should be deterministic.
5. Attention in the latency demo runs as one partition at batch size 1, so there is no split-KV merge.
6. The `throughput` branch uses tiled matmuls with full-K tiles, which should be batch-invariant per row. It is a separate program from the latency kernel, though, so switching between them would change the arithmetic.
7. The newer MegaKittens branch, `mk-v2-llama` from May 2026, requires B200 or B300 and CUDA 13.

### 3.2 Mirage Persistent Kernel (MPK, OSDI 2026)

1. MPK has task implementations for Ampere (`tasks/ampere`), Hopper and datacenter Blackwell. `linear_layer` accepts compute capability 8.0 to 8.9, and the repo includes an sm86 test, so the A40 (sm_86) and A100 (sm_80) are plausible targets.
2. MPK rejects sm_120 in `linear_layer`. Its runtime header would also select a 207 KB shared-memory budget for any capability of 9.0 or above, which consumer Blackwell cannot provide. The RTX 5060 would need a port.
3. MPK supports speculative decoding with prompt lookup, lookahead and EAGLE-3, with strict greedy verification.
4. In the Qwen3 demo, enabling spec decode switches attention from `paged_attention_layer` to `single_batch_extend_attention_layer`. Verify and decode therefore run different attention kernels. The argmax partition grid also changes with the speculation length.
5. The Ampere linear kernel has no split-K and a fixed 128-wide K loop. Linear layers are therefore likely row-invariant, which would isolate attention as the main verify-vs-decode suspect.
6. MPK's own in-repo developer notes say token identity is inconclusive on the DeepSeek-V3 TP8 path because of cross-CTA fp32 `atomicAdd` in the FFN, with runs diverging around token 10. The notes' acceptance bar is cosine similarity of at least 0.999 and relative error of at most 3e-2, not bitwise equality.

### 3.3 Production megakernels and related work

1. Cohere's H100 megakernel engine selects a prebuilt schedule per batch size and computes attention split counts from live sequence lengths. It validated accuracy by benchmark scores only, with SciCode at 38.9% versus 38.2% for vLLM.
2. LMSYS's Muse Spark Open megakernel verifies 16-token trees on GB300 and reports no accuracy or determinism evaluation.
3. An independent H100 comparison found that the Hazy megakernel misses numerical-fidelity criteria that a CUDA-graphs baseline passes, and that Hazy emits bf16 logits.
4. DeepSeek-V4 describes a dual-kernel decode attention whose two kernels are designed to be bitwise identical. That is a direct precedent for the verify-path mitigation proposed here.
5. CoRun formalizes batch invariance versus position invariance and gets determinism by padding to fixed shapes.

## 4. CPU characterization results

These are emulations, not GPU measurements. A 3.8M-parameter Llama-style character model was trained on TinyShakespeare, reaching a validation loss of 1.62. Its forward pass was then rerun with explicit reduction schedules. Storage is bf16 between ops and accumulation is fp32, as in the megakernels. Every row is computed with identical shapes, so any difference comes only from the schedule under test. The harness was validated first. The reference schedule gave bitwise-equal verify and decode logits and bitwise-equal reruns, and it agreed with the fp32 PyTorch model on 100% of top-1 tokens in a sanity run. The emulator does not model real arrival-order statistics, tensor-core accumulation order or kernel selection, so its numbers are directional only.

### 4.1 Schedules

| ID | Emulates | Policy |
|---|---|---|
| C0 | Invariant reference | 4 down-proj partials summed in fixed fp32 order, one bf16 round |
| C1 | Hazy latency kernel | 4 bf16 partials added into the bf16 residual in random arrival order, bf16 logits |
| C2 | fp32 atomics (MPK DSV3 FFN style) | Partials added into an fp32 accumulator in random order |
| M1 | Fix for C1 | C1 with a fixed-order fp32 combine |
| C3 | Shape-dependent schedule (Cohere style) | Split-K count depends on row count, and the attention split count depends on length |
| M2 | Fix for C3 | Fixed split-K counts and fixed-size attention splits |

### 4.2 Results

| Experiment | C0 | C1 | C2 | M1 | C3 | M2 |
|---|---|---|---|---|---|---|
| Run-to-run, prompts identical across 5 seeds (16 prompts, 128 tokens) | 16/16 | 2/16 | 16/16 | 16/16 | n/a | n/a |
| Run-to-run, logit rows bitwise equal, seed 0 vs 1 | 100% | 0% | 86.6% | 100% | n/a | n/a |
| Run-to-run, max logit difference | 0 | 0.119 | 0.022 | 0 | n/a | n/a |
| Long run-to-run, seed 0 vs 1 (40 prompts, 384 tokens) | n/a | n/a | 31/40 identical | n/a | n/a | n/a |
| Verify vs decode, logit rows bitwise equal at k = 1 / 3 / 7 | 100% | n/a | n/a | n/a | 97.5 / 96.5 / 95.0% | 100% |
| Spec (k=4) vs greedy, exact outputs (16 prompts, 128 tokens) | 16/16 | 7/16 | n/a | n/a | 16/16 | 16/16 |
| Long spec vs greedy (40 prompts, 384 tokens) | n/a | n/a | n/a | n/a | 29/40, first divergence 60 to 382 | n/a |
| Mean KL vs fp32 reference | 1.78e-5 | 4.43e-5 | 1.75e-5 | 3.87e-5 | n/a | n/a |
| Top-1 agreement with fp32 reference | 99.76% | 98.69% | 99.66% | 98.59% | n/a | n/a |

### 4.3 What the emulation shows

1. For determinism, accumulating partials in bf16 is far worse than accumulating them in fp32. C1 changed every logit row between seeds, while C2 left 86.6% of rows bitwise equal.
2. For fidelity, most of C1's loss comes from emitting bf16 logits rather than from the accumulation order. M1, which keeps bf16 logits but fixes the combine, still has 2.2 times C0's KL. C1 adds only about 15% more on top of that. Determinism and fidelity are therefore separate axes, and the fixes should be reported separately.
3. Small perturbations still diverge eventually. C2 and C3 showed no divergence in 128-token runs, yet 9 of 40 and 11 of 40 prompts diverged within 384 tokens. Short tests would therefore understate the problem.
4. Only 1.9% of positions had a top-2 logit margin below 0.01, and 7.9% fell below 0.05. A character model has much more peaked distributions than a 150k-token BPE model, so real models would likely diverge sooner. This is a hypothesis that the GPU runs would test.
5. Both fixes restored 100% bitwise parity in the emulator. M1 fixes run-to-run nondeterminism and M2 fixes verify-vs-decode mismatch.

### 4.4 Cost estimates for the fixes

These are roofline estimates, not measurements.

1. M1 would add about 1 MB of fp32 partial traffic per Llama-1B decode step, which is 0.04% of the 2.47 GB weight stream. Its real cost would be one extra dependency per layer, which could be folded into the next instruction's prologue.
2. M2 would run the decode kernel's arithmetic on k+1 rows and reuse each weight tile across rows. CUDA-core matvec stays memory-bound up to about 20 rows on an H100, about 54 rows on an A40 and only about 10 rows on an A100 80GB. Verify at k+1 ≤ 8 would therefore cost roughly one decode step in theory, but the A100 sits close to the compute-bound edge and would be the harder test of this fix. Real kernels reach about 78% of bandwidth, and extra register pressure would add some overhead.

## 5. Research questions and hypotheses

1. **RQ1:** Are open-source megakernels run-to-run deterministic? H1 is that the Hazy latency megakernel is not, because of bf16 `store_add` arrival order, and that a fixed-order fp32 combine fixes it at under 2% latency cost.
2. **RQ2:** Does megakernel speculative decoding reproduce plain megakernel decoding? H2 is that MPK spec output differs from MPK non-spec output on the A40 and A100, mainly because of the attention kernel switch, while its linears are row-invariant.
3. **RQ3:** Can verification be made bitwise-faithful cheaply? H3 is that routing verify and decode through one attention kernel with a fixed KV tile, or a row-replicated matvec in Hazy, restores parity with under 15% verify-step overhead at k+1 ≤ 8.
4. **RQ4:** How does divergence scale? H4 is that exact-match rate falls with generation length and rises with the precision of the accumulation target, with bf16 worst.

## 6. Platforms

| Platform | Role | Cost | Main risk |
|---|---|---|---|
| A40, sm_86, 48 GB, 84 SMs | Main MPK platform for verify-vs-decode tests, kernel attribution and fixes, with Qwen3-0.6B, Qwen3-1.7B and Qwen3-8B | Free | MPK build or end-to-end run on Ampere is unverified |
| A100, sm_80, 108 SMs | Same MPK builds and prompts for the SM-count comparison and a second Ampere data point | Free | Same as the A40 |
| Modal H100 | Hazy latency megakernel with Llama-3.2-1B, plus MPK Hopper path as a cross-check | About $50, plus $30 per month free credit, for roughly 12 to 25 H100-hours | Gated Llama weights, Nsight counters may be blocked, debugging time |
| RTX 5060, sm_120, 8 GB | Local development, reference runs and a timeboxed MPK port of about 2 days in week 3 | Free | MPK needs a shared-memory budget and task-selection port |
| CPU | The emulator in this plan | Free | Emulates schedules, not real kernels |

## 7. Experiments on real hardware

1. **E0, setup and noise floor (week 1):** The team would build MPK on the A40 and A100 and run the Qwen3 demo with and without `--use-mirage`. It would record the logit or token hashes of 5 fresh launches. The first day would be a go/no-go spike on whether MPK runs on sm_86 and sm_80.
2. **E1, MPK verify-vs-decode (week 1 to 2):** The team would compare MPK spec decode with `--spec-decode promptlookup` at k of 3 and 5 against MPK non-spec decode on Qwen3-0.6B and Qwen3-1.7B, using 64 Spec-Bench and HumanEval prompts at 512 tokens. Metrics would be exact-match rate, first-divergence position and acceptance length.
3. **E2, kernel-level attribution (week 2):** Using MPK's standalone task harnesses in `tests/runtime_python`, the team would feed identical inputs to `paged_attention` and `single_batch_extend` and compare row t bitwise. It would do the same for the linear task at batch 1 versus k+1, and for the argmax partial grids.
4. **E2b, SM-count invariance (week 2):** The team would run identical MPK builds, prompts and seeds on the A40 with 84 SMs and the A100 with 108 SMs, and compare token and logit hashes. MPK and Hazy both assign work by SM count, so this would test whether any floating-point reduction is partitioned by worker count. MPK's argmax partitions do change with worker count, but argmax is exact apart from ties.
5. **E3, Hazy run-to-run (week 2, Modal):** The team would run the Hazy latency megakernel 20 times on 32 prompts at 256 tokens and hash the logits at every step. It would then add the M1 fixed-order combine and repeat. This would be the main H100 spend, at about 4 to 6 hours.
6. **E4, mitigation in MPK (week 3):** The team would route decode and verify through one attention kernel with a fixed 64-token KV tile, either by running decode through the extend kernel with zero extra tokens or by making paged attention handle the verify rows. It would then remeasure E1 and E2.
7. **E5, costs and scaling (week 3 to 4):** The team would measure step latency for each fix, exact-match rate versus generation length, and fidelity versus HF fp32 using KL and top-1 agreement.

## 8. Timeline

| Week | Person A | Person B | Deliverable |
|---|---|---|---|
| 1 | MPK build and E0 on the A40 and A100 | Port the emulator results into the report and set up the Modal image and weights | Go/no-go on MPK, noise floor |
| 2 | E1 and E2 on MPK | E3 on Modal | Parity matrix, attributed kernels, Hazy determinism verdict |
| 3 | E4 mitigation in MPK | M1 patch for Hazy on Modal | Before-and-after parity and latency |
| 4 | E5 scaling and costs | Write-up, figures, upstream issue drafts | Final report, harness, emulator |

## 9. Risks and fallbacks

1. **MPK does not run on Ampere.** The team would fall back to MPK's Hopper path on Modal, which would roughly double H100 use, and keep the emulator as the controlled study.
2. **Hazy run-to-run turns out to be deterministic on H100.** The result would still be reported, because it would show that TMA reduce-add ordering is stable in practice. The study would then focus on MPK.
3. **The H100 budget runs out.** Development would stay local, and the cloud would run only finished batch jobs from saved images.
4. **Gated Llama-3.2-1B weights.** The team would request access on day one.
5. **Upstream reception.** Any issue or PR text would be written by hand and would lead with data.

## 10. Deliverables

1. A characterization of run-to-run, verify-vs-decode and cross-SM-count parity for two open-source megakernels on Ampere and Hopper.
2. A kernel-level attribution of every mismatch found.
3. Two mitigations with measured parity and latency cost.
4. The reduction-schedule emulator and the parity harnesses as open-source code.
5. Draft upstream issues for MPK and Hazy Megakernels.

## 11. Sources

1. HazyResearch Megakernels: https://github.com/HazyResearch/Megakernels
2. Mirage Persistent Kernel: https://github.com/mirage-project/mirage and https://www.usenix.org/conference/osdi26/presentation/cheng
3. Megakernels vs CUDA Graphs comparison: https://github.com/Pranshu-Bahadur/megakernels-vs-cuda-graphs
4. Cohere megakernel engine: https://cohere.com/blog/megakernels
5. LMSYS Muse Spark Open megakernel: https://lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast/
6. vLLM PR #63275: https://github.com/vllm-project/vllm/pull/63275
7. vLLM PR #62193: https://github.com/vllm-project/vllm/pull/62193
8. DeepSeek-V4 report: https://arxiv.org/pdf/2606.19348
9. CoRun: https://arxiv.org/pdf/2608.14376
10. Together AI on CUDA Graphs plus PDL: https://www.together.ai/blog/road-to-together-compiler
11. Modal pricing: https://modal.com/pricing
