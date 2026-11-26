"""Classify each divergence between two arms as a near-tie or something larger.

For every prompt where arm A and arm B first emit different tokens, this
script replays the shared prefix through a Hugging Face fp32 eager model and
reports the fp32 top-2 tokens and their logit margin at that position. A
numerical divergence should land on a small fp32 margin, with both branch
tokens among the fp32 top candidates. A large margin, or a branch token far
down the fp32 ranking, points at a logic bug rather than rounding.

  python experiments/margins/margin_at_divergence.py \\
      --runs results/mpk/NVIDIA_A40/Qwen_Qwen3-1.7B --a ns --b spec_k3 \\
      --model Qwen/Qwen3-1.7B --out results/margins/a40_q17_ns_vs_k3.json

The prompt is rebuilt with the same chat template MPK's demo uses, and the
token count is checked against each run's recorded prompt_length.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from mkparity.analysis import load_runs  # noqa: E402
from mkparity.bits import first_divergence  # noqa: E402

MPK_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def prompt_ids(tok, prompt: str, system: str | None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok([text], return_tensors="pt").input_ids[0].tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="directory holding the arm subdirectories")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=str(REPO / "data" / "prompts" / "eval64.jsonl"))
    ap.add_argument("--system", default=MPK_SYSTEM, help="system prompt; pass '' for none (Hazy)")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = {json.loads(l)["prompt_id"]: json.loads(l)["prompt"]
               for l in Path(args.prompts).read_text().splitlines() if l.strip()}
    runs = load_runs(args.runs)
    pick = lambda arm: {r["prompt_id"]: r for r in runs
                        if r["arm"] == arm and r.get("rep", 0) == args.rep and r.get("returncode", 0) == 0}
    A, B = pick(args.a), pick(args.b)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="eager").to(device).eval()
    rows = []
    for pid in sorted(set(A) & set(B)):
        ta, tb = A[pid]["token_ids"], B[pid]["token_ids"]
        f = first_divergence(ta, tb)
        if f < 0 or f >= min(len(ta), len(tb)):
            continue
        pids = prompt_ids(tok, prompts[pid], args.system or None)
        rec_len = A[pid].get("prompt_length")
        with torch.no_grad():
            ids = torch.tensor([pids + ta[:f]], device=device)
            logits = model(ids).logits[0, -1].float()
        top = torch.topk(logits, 5)
        rank = lambda t: int((logits > logits[t]).sum())
        rows.append(dict(
            prompt_id=pid, first_divergence=f, prompt_length_matches=(rec_len is None or rec_len == len(pids)),
            token_a=ta[f], token_b=tb[f], fp32_rank_a=rank(ta[f]), fp32_rank_b=rank(tb[f]),
            fp32_top5=top.indices.tolist(), fp32_top2_margin=float(top.values[0] - top.values[1]),
            fp32_gap_a_b=float(abs(logits[ta[f]] - logits[tb[f]]))))
        print(json.dumps(rows[-1]), flush=True)
    near = sum(1 for r in rows if r["fp32_rank_a"] <= 1 and r["fp32_rank_b"] <= 1)
    out = dict(a=args.a, b=args.b, model=args.model, divergences=len(rows),
               both_in_fp32_top2=near, rows=rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} divergences, {near} with both branch tokens in the fp32 top 2")


if __name__ == "__main__":
    main()
