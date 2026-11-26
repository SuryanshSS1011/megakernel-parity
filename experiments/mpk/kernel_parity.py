"""E2: kernel-level parity checks for MPK tasks, built with PersistentKernel
test mode. Needs a CUDA GPU and an installed mirage.

Each check builds the same task several times with a different row count or
task count and asks whether a given row comes out bit-identical. Shapes match
Qwen3-1.7B (hidden 2048, intermediate 6144, 16 query heads, 8 KV heads,
head_dim 128).

  python experiments/mpk/kernel_parity.py --out results/mpk_kernels/$(hostname).json
  python experiments/mpk/kernel_parity.py --only linear

Checks:
  linear       linear_layer and linear_with_residual_layer at M = 1, 2, 4, 6, 8
               versus M = 8, row by row
  lm_head      the vocab projection split into 64 tasks (A40 worker count) and
               96 tasks (A100 worker count) on the same GPU
  attention    paged_attention_layer causal prefill at n = 1, 2, 4, 8, 16 tokens;
               row i of prefill(n) versus row i of prefill(16)

Not covered yet: single_batch_extend_attention, the spec-decode attention
path. It reads a non-paged [batch, seq, kv_heads, head_dim] cache and takes its
query count from the input rows, and its sequence position comes from runtime
step metadata, so it should be written on the GPU box once the smoke run shows
how test mode feeds that metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from mkparity.bits import bitwise_equal  # noqa: E402

DT = torch.bfloat16
HIDDEN, INTER, QH, KVH, HD, VOCAB = 2048, 6144, 16, 8, 128, 153600


def make_pk(tokens, requests, **extra):
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel

    workers, scheds = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    meta = extra.pop("meta_tensors", None)
    params.update(test_mode=True, num_workers=workers, num_local_schedulers=scheds, mpi_rank=0,
                  world_size=1, max_num_batched_tokens=tokens, max_num_batched_requests=requests, **extra)
    if meta is not None:
        params["meta_tensors"] = meta
    return PersistentKernel(**params)


def block_dim(pk):
    return (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)


def linear_grid(n):
    return n // 96 if n % 96 == 0 else n // 64


def run_pk(pk):
    with tempfile.TemporaryDirectory() as d:
        pk.compile(output_dir=d)
        pk()
        torch.cuda.synchronize()
    pk.finalize()


def row_report(ref, out, rows):
    eq = [bitwise_equal(out[i], ref[i]) for i in range(rows)]
    diff = (out[:rows].float() - ref[:rows].float()).abs().max().item()
    return dict(rows=rows, rows_bitwise_equal=sum(eq), all_equal=all(eq), max_abs_diff=diff)


def check_linear(g):
    """Same weights and rows, different M. Returns per-layer, per-M reports."""
    X2 = torch.randn(8, HIDDEN, dtype=DT, device="cuda", generator=g)
    X6 = torch.randn(8, INTER, dtype=DT, device="cuda", generator=g)
    R = torch.randn(8, HIDDEN, dtype=DT, device="cuda", generator=g)
    W = {
        "qkv": torch.randn((QH + 2 * KVH) * HD, HIDDEN, dtype=DT, device="cuda", generator=g) * 0.02,
        "o_proj": torch.randn(HIDDEN, QH * HD, dtype=DT, device="cuda", generator=g) * 0.02,
        "gate_up": torch.randn(2 * INTER, HIDDEN, dtype=DT, device="cuda", generator=g) * 0.02,
        "down_res": torch.randn(HIDDEN, INTER, dtype=DT, device="cuda", generator=g) * 0.02,
    }
    outs = {}
    for M in (8, 1, 2, 4, 6):
        pk = make_pk(M, M)
        bd = block_dim(pk)
        res = {}
        for name, w in W.items():
            n = w.shape[0]
            o = torch.zeros(M, n, dtype=DT, device="cuda")
            wdt = pk.attach_input(w, name=f"w_{name}")
            odt = pk.attach_input(o, name=f"o_{name}")
            if name == "down_res":
                pk.linear_with_residual_layer(
                    input=pk.attach_input(X6[:M].clone(), name="x6"), weight=wdt,
                    residual=pk.attach_input(R[:M].clone(), name="res"), output=odt,
                    grid_dim=(HIDDEN // 64, 1, 1), block_dim=bd)
            else:
                xin = X2 if w.shape[1] == HIDDEN else X6
                pk.linear_layer(input=pk.attach_input(xin[:M].clone(), name=f"x_{name}"),
                                weight=wdt, output=odt, grid_dim=(linear_grid(n), 1, 1), block_dim=bd)
            res[name] = o
        run_pk(pk)
        outs[M] = {k: v.clone() for k, v in res.items()}
    report = {}
    for name in W:
        ref = outs[8][name]
        fp32 = (X6 if name == "down_res" else X2).float() @ W[name].float().T
        if name == "down_res":
            fp32 = fp32 + R.float()
        report[name] = dict(
            kernel_vs_fp32_max_abs=(ref.float() - fp32).abs().max().item(),
            by_M={str(M): row_report(ref, outs[M][name], M) for M in (1, 2, 4, 6)},
        )
    return report


def check_lm_head(g):
    X = torch.randn(8, HIDDEN, dtype=DT, device="cuda", generator=g)
    W = torch.randn(VOCAB, HIDDEN, dtype=DT, device="cuda", generator=g) * 0.02
    pk = make_pk(8, 8)
    bd = block_dim(pk)
    xdt, wdt = pk.attach_input(X, name="x"), pk.attach_input(W, name="lm_head")
    outs = {}
    for tasks in (64, 96):
        o = torch.zeros(8, VOCAB, dtype=DT, device="cuda")
        pk.linear_layer(input=xdt, weight=wdt, output=pk.attach_input(o, name=f"o{tasks}"),
                        grid_dim=(tasks, 1, 1), block_dim=bd)
        outs[tasks] = o
    run_pk(pk)
    r = row_report(outs[96], outs[64], 8)
    r["argmax_equal"] = bool(torch.equal(outs[64].argmax(-1), outs[96].argmax(-1)))
    return {"tasks_64_vs_96": r}


def check_attention(g):
    n_max, page, max_seq = 16, 64, 256
    qkv = torch.randn(n_max, (QH + 2 * KVH) * HD, dtype=DT, device="cuda", generator=g)
    outs = {}
    for n in (n_max, 1, 2, 4, 8):
        pk = make_pk(n, 1, max_seq_length=max_seq, max_num_pages=4, page_size=page,
                     meta_tensors={"prompt_lengths": torch.tensor([n], dtype=torch.int32, device="cuda")})
        cos = torch.ones(max_seq, HD, dtype=DT, device="cuda")
        sin = torch.zeros(max_seq, HD, dtype=DT, device="cuda")
        norm = torch.ones(HD, dtype=DT, device="cuda")
        kc = torch.zeros(4, page, KVH, HD, dtype=DT, device="cuda")
        out = torch.zeros(n, QH * HD, dtype=DT, device="cuda")
        nd = pk.attach_input(norm, name="norm")
        pk.paged_attention_layer(
            input=pk.attach_input(qkv[:n].clone(), name="qkv"),
            k_cache=pk.attach_input(kc, name="k_cache"),
            v_cache=pk.attach_input(torch.zeros_like(kc), name="v_cache"),
            q_norm=nd, k_norm=nd,
            cos_pos_embed=pk.attach_input(cos, name="cos"), sin_pos_embed=pk.attach_input(sin, name="sin"),
            output=pk.attach_input(out, name="out"),
            grid_dim=(1, KVH, 1), block_dim=block_dim(pk), enable_qk_norm=False)
        run_pk(pk)
        outs[n] = out.clone()
    ref = outs[n_max]
    return {f"prefill_{n}_vs_{n_max}": row_report(ref, outs[n], n) for n in (1, 2, 4, 8)}


CHECKS = {"linear": check_linear, "lm_head": check_lm_head, "attention": check_attention}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(CHECKS), action="append")
    ap.add_argument("--out", default=str(REPO / "results" / "mpk_kernels" / "kernel_parity.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("kernel_parity.py needs a CUDA GPU")
    p = torch.cuda.get_device_properties(0)
    report = dict(gpu=dict(name=p.name, capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count),
                  seed=args.seed, checks={})
    for name in args.only or list(CHECKS):
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        try:
            report["checks"][name] = CHECKS[name](g)
        except Exception as e:  # keep going so one broken check does not hide the others
            report["checks"][name] = dict(error=f"{type(e).__name__}: {e}")
        print(name, json.dumps(report["checks"][name])[:400], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
