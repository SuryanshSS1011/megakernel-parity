"""Verify(M=k+1) vs decode(M=1) parity harness for a Hugging Face causal LM.

For each prompt it builds the model's own greedy continuation, then treats the
next k+1 tokens as a fully accepted speculative window. The same k+1 tokens are
scored two ways from an identical prefix KV cache:

  decode: k+1 sequential forwards with one query row each (plain greedy decode)
  verify: one forward with k+1 query rows (the spec-dec verification pass)

Rows are compared bitwise. On a mismatch the harness reports the first module
(in forward order) whose output differs, with per-layer taps aligned by row.

Usage (GPU):
  python -I parity_hf.py --model Qwen/Qwen3-0.6B --k 1 2 4 7 --bi none
  python -I parity_hf.py --model Qwen/Qwen3-0.6B --k 4 --bi vllm --attn eager
  python -I parity_hf.py --model Qwen/Qwen3-0.6B --k 4 --dtype float32
"""

import argparse
import copy
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "serving " * 48,
    "def fibonacci(n):\n    ",
    "The three main causes of the French Revolution were",
    "Translate to French: The weather is nice today.",
]

# Fine-grained taps so a mismatch lands on one op: projections (GEMMs),
# q/k norms, the attention block, norms and the MLP GEMMs.
TAP_SUFFIXES = (
    "input_layernorm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
    "self_attn", "o_proj", "post_attention_layernorm",
    "gate_proj", "up_proj", "down_proj", "mlp", "lm_head",
)


def as_bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a float tensor as same-width ints so == is a bitwise test."""
    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(width)


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool((as_bits(a) == as_bits(b)).all())


class Tap:
    """Records module outputs per forward call, in forward order."""

    def __init__(self, model):
        self.calls: dict[str, list[torch.Tensor]] = {}
        self.order: list[str] = []
        self.handles = []
        for name, mod in model.named_modules():
            if name.endswith(TAP_SUFFIXES) or name.endswith("model.norm"):
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def f(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if name not in self.calls:
                self.calls[name] = []
                self.order.append(name)
            self.calls[name].append(o.detach().clone())
        return f

    def rows(self, name: str) -> torch.Tensor:
        # Decode calls each produce [1, 1, H]; verify produces one [1, k+1, H].
        return torch.cat(self.calls[name], dim=1)

    def clear(self):
        self.calls, self.order = {}, []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def prefill(model, prefix_ids):
    out = model(prefix_ids, use_cache=True)
    return out.past_key_values


@torch.no_grad()
def decode_rows(model, cache, window_ids):
    """Plain decode: one query row per forward. Returns logits [1, k+1, V]."""
    rows = []
    for j in range(window_ids.shape[1]):
        o = model(window_ids[:, j : j + 1], past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        rows.append(o.logits[:, -1])
    return torch.stack(rows, 1)


@torch.no_grad()
def verify_rows(model, cache, window_ids):
    """Spec-dec verify: all k+1 rows in one forward. Returns logits [1, k+1, V]."""
    return model(window_ids, past_key_values=cache, use_cache=True).logits


def compare_logits(ref: torch.Tensor, ver: torch.Tensor) -> dict:
    ref, ver = ref[0], ver[0]  # [k+1, V]
    row_eq = (as_bits(ref) == as_bits(ver)).all(-1)
    diff = (ref.float() - ver.float()).abs().amax(-1)
    top2 = ref.float().topk(2, -1).values
    first_bad = int((~row_eq).nonzero()[0]) if (~row_eq).any() else -1
    return dict(
        bitwise_rows=row_eq.tolist(),
        first_divergent_row=first_bad,
        max_abs=[round(x, 8) for x in diff.tolist()],
        top2_margin=[round(x, 6) for x in (top2[:, 0] - top2[:, 1]).tolist()],
        argmax_flip=(ref.argmax(-1) != ver.argmax(-1)).tolist(),
    )


def localize(model, cache, window_ids):
    """Rerun both paths with taps and return the first module that differs."""
    tap = Tap(model)
    try:
        decode_rows(model, copy.deepcopy(cache), window_ids)
        dec = {n: tap.rows(n) for n in tap.order}
        order = list(tap.order)
        tap.clear()
        verify_rows(model, copy.deepcopy(cache), window_ids)
        ver = {n: tap.rows(n) for n in tap.order}
    finally:
        tap.remove()
    for name in order:
        if not bitwise_equal(dec[name], ver[name]):
            d = (dec[name].float() - ver[name].float()).abs()
            row_bad = (as_bits(dec[name]) != as_bits(ver[name])).flatten(2).any(-1)[0]
            return dict(module=name, max_abs=float(d.max()), rows_differing=row_bad.tolist())
    return None


def enable_bi(mode: str):
    if mode == "tm":
        # Covers mm, addmm, log_softmax and mean.dim only. Attention (bmm, softmax,
        # SDPA) is NOT covered, so attention divergence will remain.
        from batch_invariant_ops import set_batch_invariant_mode
        ctx = set_batch_invariant_mode(True)
        ctx.__enter__()
        return ctx
    if mode == "vllm":
        # Covers mm/addmm/matmul/linear (Triton persistent on SM8x), bmm, softmax,
        # log_softmax and mean.dim. Pair with --attn eager so attention runs on bmm.
        from vllm.model_executor.determinism.batch_invariant import enable_batch_invariant_mode
        enable_batch_invariant_mode()
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 7])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--bi", default="none", choices=["none", "tm", "vllm"])
    p.add_argument("--offsets", type=int, nargs="+", default=[0, 13, 31, 64],
                   help="window start offsets into the greedy continuation")
    p.add_argument("--gen", type=int, default=96, help="greedy tokens to generate per prompt")
    p.add_argument("--prompts-file", default=None, help="one prompt per line")
    p.add_argument("--out", default="parity.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-localize", action="store_true")
    args = p.parse_args(argv)

    enable_bi(args.bi)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn).to(args.device).eval()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]

    summary = dict(windows=0, bitwise_windows=0, rows=0, bitwise_rows=0, flips=0)
    with open(args.out, "w") as fout:
        for pi, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
            # Greedy continuation from plain M=1 decode is the reference trajectory.
            cache = prefill(model, ids[:, :-1])
            cont = [ids[:, -1:]]
            for _ in range(args.gen):
                o = model(cont[-1], past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                cont.append(o.logits[:, -1:].argmax(-1))
            full = torch.cat([ids[:, :-1]] + cont, dim=1)
            n_prompt = ids.shape[1] - 1

            for k in args.k:
                for off in args.offsets:
                    start = n_prompt + off
                    if start + k + 1 > full.shape[1]:
                        continue
                    prefix, window = full[:, :start], full[:, start : start + k + 1]
                    base = prefill(model, prefix)
                    ref = decode_rows(model, copy.deepcopy(base), window)
                    ref2 = decode_rows(model, copy.deepcopy(base), window)
                    ver = verify_rows(model, copy.deepcopy(base), window)
                    ver2 = verify_rows(model, copy.deepcopy(base), window)
                    rec = dict(prompt_idx=pi, k=k, M=k + 1, prefix_len=start,
                               dtype=args.dtype, attn=args.attn, bi=args.bi,
                               rerun_decode_bitwise=bitwise_equal(ref, ref2),
                               rerun_verify_bitwise=bitwise_equal(ver, ver2),
                               **compare_logits(ref, ver))
                    if rec["first_divergent_row"] >= 0 and not args.no_localize:
                        rec["first_divergent_module"] = localize(model, base, window)
                    fout.write(json.dumps(rec) + "\n")
                    summary["windows"] += 1
                    summary["bitwise_windows"] += int(all(rec["bitwise_rows"]))
                    summary["rows"] += len(rec["bitwise_rows"])
                    summary["bitwise_rows"] += sum(rec["bitwise_rows"])
                    summary["flips"] += sum(rec["argmax_flip"])
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
