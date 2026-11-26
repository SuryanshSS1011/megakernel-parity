"""Summarize parity runs into a markdown table and a JSON report.

  python scripts/analyze.py results/ --out results/summary

Comparisons made when the data exist:
  1. MPK: every arm against the plain-decode reference `ns`, per GPU and model.
  2. MPK: `ns_rep` repetitions against repetition 0 (run-to-run).
  3. MPK: each row of the 4-request batch `ns_B4` against `ns` (batch
     invariance) and against row 0 of the same batch (position invariance).
  4. MPK: `ns` on one GPU against `ns` on another for the same model (cross-GPU).
  5. Hazy: every megakernel repetition against repetition 0, on tokens and on
     per-step logit hashes, and the megakernel against the torch decode.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mkparity.analysis import compare_hashes, compare_streams, load_runs, markdown_table  # noqa: E402


def index(runs):
    """{(system, gpu, model): {arm: {rep: {prompt_id: run}}}}"""
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in runs:
        if r.get("returncode", 0) != 0 or not r.get("token_ids"):
            continue
        key = (r["system"], r.get("gpu", {}).get("name"), r.get("model"))
        idx[key][r["arm"]][r.get("rep", 0)][r["prompt_id"]] = r
    return idx


def streams(d, field="token_ids", row=None):
    if row is None:
        return {pid: r[field] for pid, r in d.items() if r.get(field)}
    return {pid: r["all_token_ids"][row] for pid, r in d.items()
            if r.get("all_token_ids") and len(r["all_token_ids"]) > row}


def analyze(runs):
    idx = index(runs)
    rows, failures = [], defaultdict(int)
    for r in runs:
        if r.get("returncode", 0) != 0:
            failures[(r["system"], r.get("arm"))] += 1
    for (system, gpu, model), arms in sorted(idx.items(), key=lambda kv: str(kv[0])):
        tag = f"{system} {gpu} {model}"
        if system == "mpk" and "ns" in arms:
            ref = streams(arms["ns"][0])
            for arm in sorted(arms):
                if arm in ("ns", "ns_rep", "ns_B4"):
                    continue
                rows.append(dict(name=f"{tag}: {arm} vs ns", result=compare_streams(ref, streams(arms[arm][0]))))
            if "ns_rep" in arms:
                r0 = streams(arms["ns_rep"][0])
                for rep in sorted(arms["ns_rep"])[1:]:
                    rows.append(dict(name=f"{tag}: ns_rep r{rep} vs r0",
                                     result=compare_streams(r0, streams(arms["ns_rep"][rep]))))
            if "ns_B4" in arms:
                b = arms["ns_B4"][0]
                n_rows = max((len(x.get("all_token_ids") or []) for x in b.values()), default=0)
                for row in range(n_rows):
                    rows.append(dict(name=f"{tag}: ns_B4 row {row} vs ns",
                                     result=compare_streams(ref, streams(b, row=row))))
                    if row:
                        rows.append(dict(name=f"{tag}: ns_B4 row {row} vs row 0",
                                         result=compare_streams(streams(b, row=0), streams(b, row=row))))
        if system == "hazy" and "mk" in arms:
            reps = arms["mk"]
            r0 = streams(reps[0])
            h0 = streams(reps[0], "logit_hashes")
            for rep in sorted(reps)[1:]:
                res = compare_streams(r0, streams(reps[rep]))
                res["logit_hashes"] = compare_hashes(h0, streams(reps[rep], "logit_hashes"))
                rows.append(dict(name=f"{tag}: mk r{rep} vs r0", result=res))
            if "torch" in arms:
                rows.append(dict(name=f"{tag}: mk r0 vs torch",
                                 result=compare_streams(streams(arms["torch"][0]), r0)))
    by_model = defaultdict(dict)
    for (system, gpu, model), arms in idx.items():
        if system == "mpk" and "ns" in arms:
            by_model[model][gpu] = streams(arms["ns"][0])
    for model, gpus in by_model.items():
        for g1, g2 in combinations(sorted(gpus), 2):
            rows.append(dict(name=f"mpk {model}: ns on {g2} vs {g1}",
                             result=compare_streams(gpus[g1], gpus[g2])))
    return rows, {f"{s} {a}": n for (s, a), n in failures.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=str(REPO / "results" / "summary"))
    args = ap.parse_args(argv)
    runs = [r for root in args.roots for r in load_runs(root)]
    rows, failures = analyze(runs)
    md = f"# Parity summary\n\n{len(runs)} run files loaded.\n\n" + markdown_table(rows)
    if failures:
        md += "\n## Failed launches\n\n" + "\n".join(f"1. {k}: {v}" for k, v in sorted(failures.items())) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md)
    out.with_suffix(".json").write_text(json.dumps(dict(rows=rows, failures=failures), indent=1))
    print(md)


if __name__ == "__main__":
    main()
