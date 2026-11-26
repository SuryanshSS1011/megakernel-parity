"""Build the fixed 64-prompt evaluation set from public sources.

The repo stores only the selected ids (prompts/eval64_ids.json). This script
downloads Spec-Bench and HumanEval from GitHub, checks their sha256, and
writes the materialized prompts as JSONL.

  python scripts/build_prompts.py              # writes data/prompts/eval64.jsonl
  python scripts/build_prompts.py --select     # re-runs the id selection

Sources:
  Spec-Bench  https://github.com/hemingkx/Spec-Bench  (data/spec_bench/question.jsonl)
  HumanEval   https://github.com/openai/human-eval    (data/HumanEval.jsonl.gz)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "prompts" / "eval64_ids.json"
SPEC_URL = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl"
HE_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MT_BENCH = {"writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"}
TASKS = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag"]
PER_TASK, N_HUMANEVAL, MAX_CHARS, SEED = 9, 10, 3600, 597


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_sources():
    spec_raw, he_raw = fetch(SPEC_URL), fetch(HE_URL)
    spec = [json.loads(line) for line in spec_raw.decode().splitlines() if line.strip()]
    he = [json.loads(line) for line in gzip.open(io.BytesIO(he_raw), "rt")]
    return spec, he, {"spec_bench_sha256": sha(spec_raw), "humaneval_sha256": sha(he_raw)}


def task_of(q):
    return "mt_bench" if q["category"] in MT_BENCH else q["category"]


def select(spec, he):
    rng = random.Random(SEED)
    chosen = []
    for task in TASKS:
        pool = [q for q in spec if task_of(q) == task and len(q["turns"][0]) <= MAX_CHARS]
        pool.sort(key=lambda q: q["question_id"])
        chosen += [("spec_bench", task, q["question_id"]) for q in rng.sample(pool, PER_TASK)]
    he_ids = sorted(rng.sample([h["task_id"] for h in he], N_HUMANEVAL))
    chosen += [("humaneval", "code", t) for t in he_ids]
    return chosen


def materialize(spec, he, chosen):
    by_q = {q["question_id"]: q for q in spec}
    by_t = {h["task_id"]: h for h in he}
    rows = []
    for i, (src, task, sid) in enumerate(chosen):
        if src == "spec_bench":
            text = by_q[sid]["turns"][0]
        else:
            text = "Complete the following Python function.\n\n" + by_t[sid]["prompt"]
        rows.append(dict(prompt_id=f"p{i:02d}_{task}", source=src, task=task, source_id=sid, prompt=text))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "prompts" / "eval64.jsonl"))
    ap.add_argument("--select", action="store_true", help="recompute and overwrite the id list")
    args = ap.parse_args()
    spec, he, hashes = load_sources()
    if args.select or not IDS.exists():
        chosen = select(spec, he)
        IDS.parent.mkdir(parents=True, exist_ok=True)
        IDS.write_text(json.dumps(dict(seed=SEED, max_chars=MAX_CHARS, **hashes, items=chosen), indent=1))
    meta = json.loads(IDS.read_text())
    for k, v in hashes.items():
        if meta.get(k) != v:
            print(f"warning: {k} changed upstream ({meta.get(k)} -> {v}); ids may point at edited prompts")
    rows = materialize(spec, he, [tuple(x) for x in meta["items"]])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
