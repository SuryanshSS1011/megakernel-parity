# Evaluation Framework

The study asks one question. When a megakernel should, in exact arithmetic, produce the same output twice, does it actually produce the same bits, and if not, where does it break and what does fixing it cost?

## 1. Comparisons

Each comparison has a test arm and a reference arm. The build, model, prompts and settings would be held fixed across the two arms.

| Axis | Test arm | Reference arm | Systems |
|---|---|---|---|
| Run-to-run | Launch *i* of a build | Launch 0 of the same build | Hazy on H100, MPK on A40 and A100 |
| Verify vs decode | Spec-decode output at k = 1, 3, 5, 7 | The same megakernel's plain greedy decode | MPK on A40 and A100 |
| Cross-GPU | MPK on the A100 (108 SMs) | MPK on the A40 (84 SMs) | MPK |
| Fidelity | Megakernel logits | Hugging Face fp32 eager model | All |
| Cost | Build with a fix applied | Unmodified build | All |

## 2. Metrics

Metrics would be measured at four levels, from coarse to fine.

1. **Token level:** The exact-match rate of full outputs, the position of the first differing token, and a survival curve showing the fraction of prompts still identical at each token position. The emulator showed why the curve matters, because outputs that matched at 128 tokens diverged by 384.
2. **Logit level:** The share of logit rows that are bit-identical, the maximum absolute logit difference, the difference in ULPs, the argmax flip rate and the top-2 margin at each flip. Logits catch divergence long before tokens do.
3. **Kernel level:** For every mismatch, the first layer and then the first kernel whose output differs. Attribution comes from this level.
4. **Speculation quality and cost:** Acceptance rate, mean accepted length, decode step latency, tokens per second and the verify step's overhead relative to one decode step.

Fidelity would be reported separately as KL divergence and top-1 agreement against the fp32 reference. The emulator showed that determinism and fidelity are independent, so a fix could improve one without the other.

## 3. Workload

1. **Models:** Qwen3-0.6B, Qwen3-1.7B and Qwen3-8B would run on MPK, and Llama-3.2-1B-Instruct would run on Hazy. Llama-3.2-1B-Instruct and Qwen3-1.7B would also run on stock Hugging Face kernels as the baseline.
2. **Prompts:** Each configuration would use 64 prompts drawn from Spec-Bench's six tasks and HumanEval.
3. **Decoding:** Decoding would be greedy at temperature 0 with 512 new tokens, and spec decode would use prompt-lookup drafting.
4. **Repeats:** Run-to-run tests would use 20 fresh launches on Hazy and 5 on MPK, each in a separate process.

## 4. Protocol

1. **Noise floor first:** Before any A/B comparison, the team would record how much repeated launches of an unmodified build vary. A difference would count only if it exceeds that floor.
2. **Pinned builds:** Every result would record the commit, compiler version, driver version and GPU.
3. **Per-step hashing:** Each decode step would write a hash of its logits, so two runs could be compared without storing full tensors.
4. **Localization on mismatch:** A mismatch would trigger a layer-by-layer and then kernel-by-kernel search using MPK's standalone task harnesses.
5. **Statistics:** Proportions would carry 95% Wilson confidence intervals. With 64 prompts, a 50% rate is accurate to about ±12 points.

## 5. Pass and fail criteria

1. **Parity:** A configuration would pass only if 100% of logit rows are bit-identical across all launches and prompts. Anything less would count as a failure, however small the difference.
2. **Fix M1, the fixed-order sum:** It would succeed if it brings Hazy to 100% run-to-run parity at under 2% step-latency cost.
3. **Fix M2, the shared verify path:** It would succeed if it brings MPK to 100% verify-vs-decode parity at under 15% verify-step overhead for k+1 ≤ 8.
4. **Fidelity guard:** Neither fix could make KL against fp32 worse than the unmodified build.

## 6. Hypotheses and the experiments that test them

| Hypothesis | Tested by | Rejected if |
|---|---|---|
| H1: Hazy is not run-to-run deterministic | E3 | 20 launches are bit-identical |
| H2: MPK spec decode diverges from plain decode | E1 and E2 | 100% parity at every k |
| H3: A shared verify path fixes it cheaply | E4 and E5 | Parity stays below 100% or overhead exceeds 15% |
| H4: Divergence grows with length and is worst for bf16 accumulation | E5 and the emulator | The survival curves are flat |
| SM-count invariance | E2b | Not applicable, because this test is exploratory |

Experiment IDs refer to section 7 of [plan.md](plan.md).

## 7. Mapping to the repo

1. **pytest** checks invariants such as bit-equality helpers and the invariance of the reference schedules. CPU tests run on every commit, and GPU tests would carry the `gpu` marker.
2. **Experiment scripts** in `experiments/` run the sweeps and write JSON to `results/`.
3. **The emulator** in `src/mkparity/` stays as the controlled baseline, because the cause of every difference there is known.
