"""Back-of-envelope roofline for the two mitigations. These are planning
estimates, not measurements.

M1 deterministic combine: write S fp32 partials per split-K op and reduce them in
a fixed order, instead of S bf16 store_adds into the residual.
M2 row-replicated verify: run the decode (matvec) kernel's arithmetic on k+1
rows, reusing each weight tile across rows. While R stays under the
compute/bandwidth crossover the step stays memory-bound.
"""
import json

GPUS = {
    # name: (HBM bytes/s, fp32 CUDA-core FLOP/s, bf16 dense tensor FLOP/s)
    "H100 SXM": (3.35e12, 67e12, 989e12),
    "A100 80GB SXM": (2.039e12, 19.5e12, 312e12),
    "A40": (0.696e12, 37.4e12, 149.7e12),
}
MODELS = {
    # name: (params read per decode step, hidden, intermediate, layers, splitk_down)
    "Llama-3.2-1B": (1.236e9, 2048, 8192, 16, 4),
    "Qwen3-1.7B": (1.72e9, 2048, 6144, 28, 3),
}
rows = []
for g, (bw, f32, tc) in GPUS.items():
    for m, (P, h, inter, L, S) in MODELS.items():
        t_mem = 2 * P / bw
        m1_bytes = L * S * h * 4 * 2  # write + read fp32 partials
        r_star_cuda = f32 / bw  # rows before CUDA-core matvec becomes compute-bound
        r_star_tc = tc / bw
        verify = {}
        for R in (1, 2, 4, 8):
            t_cuda = 2 * P * R / f32
            verify[R] = round(max(t_mem, t_cuda) / t_mem, 3)
        rows.append(dict(gpu=g, model=m, weight_stream_ms=round(t_mem * 1e3, 3),
                         m1_extra_bytes=m1_bytes, m1_extra_traffic_pct=round(100 * m1_bytes / (2 * P), 4),
                         crossover_rows_cuda_core=round(r_star_cuda, 1),
                         crossover_rows_tensor_core=round(r_star_tc, 1),
                         m2_step_time_ratio_vs_decode_cuda_core=verify))
for r in rows:
    print(json.dumps(r))
json.dump(rows, open("results/emulator/cost_model.json", "w"), indent=1)
