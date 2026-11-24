"""End-to-end spec vs non-spec parity check on vLLM, bitwise on logprobs.

Run each arm in its own process (two engines in one process share GPU state),
then compare. Each arm stores emitted token ids and the exact bits of each
emitted token's logprob (float.hex), so "same tokens, different numbers" is
visible even when greedy text matches.

  # noise floor: same arm, two fresh processes
  python -I parity_vllm.py run --arm ref --out ref_a.json
  python -I parity_vllm.py run --arm ref --out ref_b.json
  # spec arms
  python -I parity_vllm.py run --arm ngram --k 4 --out ngram4.json
  python -I parity_vllm.py run --arm draft --k 4 --target Qwen/Qwen3-1.7B --out draft4.json
  # with VLLM_BATCH_INVARIANT=1 in the environment for the BI arms
  python -I parity_vllm.py compare ref_a.json ngram4.json
"""

import argparse
import json
import sys


def build_spec(arm: str, k: int, draft: str):
    if arm == "ref":
        return None
    if arm == "ngram":
        return {"method": "ngram", "num_speculative_tokens": k,
                "prompt_lookup_min": 2, "prompt_lookup_max": 5}
    if arm == "draft":
        return {"method": "draft_model", "model": draft, "num_speculative_tokens": k}
    if arm == "eagle3":
        return {"method": "eagle3", "model": draft, "num_speculative_tokens": k}
    raise ValueError(arm)


def run(args):
    import os
    from vllm import LLM, SamplingParams

    prompts = [line.rstrip("\n") for line in open(args.prompts) if line.strip()]
    kwargs = dict(
        model=args.target,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=args.eager,
        enable_prefix_caching=False,
        seed=0,
    )
    spec = build_spec(args.arm, args.k, args.draft)
    if spec is not None:
        kwargs["speculative_config"] = spec
    if args.attn:
        kwargs["attention_config"] = {"backend": args.attn}
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1, seed=0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    recs = []
    for o in outs:
        c = o.outputs[0]
        lps = [c.logprobs[i][t].logprob.hex() for i, t in enumerate(c.token_ids)]
        recs.append(dict(tokens=list(c.token_ids), logprob_hex=lps))
    meta = dict(arm=args.arm, k=args.k, target=args.target, draft=args.draft,
                eager=args.eager, attn=args.attn, max_num_seqs=args.max_num_seqs,
                batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "0"))
    json.dump(dict(meta=meta, records=recs), open(args.out, "w"))


def compare(a_path: str, b_path: str):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    n = len(a["records"])
    tok_match = lp_match = 0
    first_tok, first_lp = [], []
    for ra, rb in zip(a["records"], b["records"]):
        ft = next((i for i, (x, y) in enumerate(zip(ra["tokens"], rb["tokens"])) if x != y), -1)
        if ft == -1 and len(ra["tokens"]) != len(rb["tokens"]):
            ft = min(len(ra["tokens"]), len(rb["tokens"]))
        # Logprob bits are only comparable on the shared token prefix.
        upto = ft if ft >= 0 else len(ra["tokens"])
        fl = next((i for i in range(upto) if ra["logprob_hex"][i] != rb["logprob_hex"][i]), -1)
        tok_match += ft == -1
        lp_match += ft == -1 and fl == -1
        first_tok.append(ft)
        first_lp.append(fl)
    res = dict(a=a["meta"], b=b["meta"], prompts=n,
               token_exact=tok_match, token_exact_rate=tok_match / n,
               logprob_bitwise=lp_match, logprob_bitwise_rate=lp_match / n,
               first_token_divergence=first_tok, first_logprob_divergence=first_lp)
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=["ref", "ngram", "draft", "eagle3"])
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--target", default="Qwen/Qwen3-0.6B")
    r.add_argument("--draft", default="Qwen/Qwen3-0.6B",
                   help="draft model, or EAGLE-3 head such as AngelSlim/Qwen3-1.7B_eagle3")
    r.add_argument("--prompts", default="prompts.txt")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--max-model-len", type=int, default=4096)
    r.add_argument("--max-num-seqs", type=int, default=1)
    r.add_argument("--gpu-mem", type=float, default=0.80)
    r.add_argument("--eager", action="store_true")
    r.add_argument("--attn", default=None, help="FLASH_ATTN, TRITON_ATTN or FLEX_ATTENTION")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
