"""Analysis helpers for parity runs: exact-match rates with Wilson intervals,
first-divergence positions and divergence survival curves.

Every run file follows the `mkparity.run.v1` schema written by the runners:
a JSON object with at least `system`, `arm`, `prompt_id`, `rep` and
`token_ids`. Logit-hash runs may also carry `logit_hashes`, one per step.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

from .bits import first_divergence

CHECKPOINTS = (32, 64, 128, 256, 512)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def survival(first_divs: Iterable[int], length: int, points: Iterable[int] = CHECKPOINTS) -> dict:
    """Fraction of sequences still identical after t tokens, for each t in
    points that does not exceed length. A first divergence of -1 means the
    pair never diverged."""
    fds = list(first_divs)
    n = len(fds)
    out = {}
    for t in points:
        if t > length or n == 0:
            continue
        alive = sum(1 for f in fds if f == -1 or f >= t)
        out[t] = alive / n
    return out


def compare_streams(ref: dict, test: dict) -> dict:
    """Compare two {prompt_id: token_ids} maps over their shared prompt ids."""
    ids = sorted(set(ref) & set(test))
    fds = [first_divergence(ref[i], test[i]) for i in ids]
    n = len(ids)
    exact = sum(1 for f in fds if f == -1)
    div = sorted(f for f in fds if f >= 0)
    length = min((min(len(ref[i]), len(test[i])) for i in ids), default=0)
    lo, hi = wilson(exact, n)
    return dict(
        prompts=n,
        exact=exact,
        exact_rate=(exact / n) if n else None,
        exact_rate_ci95=[round(lo, 4), round(hi, 4)],
        first_divergence=dict(zip(ids, fds)),
        median_first_divergence=(median(div) if div else None),
        min_first_divergence=(div[0] if div else None),
        survival=survival(fds, length),
        compared_length=length,
    )


def compare_hashes(ref: dict, test: dict) -> dict:
    """Per-step logit-hash agreement between two {prompt_id: [hash, ...]} maps.
    Counts steps up to and including each stream's first differing hash."""
    ids = sorted(set(ref) & set(test))
    steps = equal = 0
    first = {}
    for i in ids:
        a, b = ref[i], test[i]
        f = first_divergence(a, b)
        first[i] = f
        m = min(len(a), len(b))
        steps += m
        equal += m if f == -1 else f
    return dict(prompts=len(ids), steps=steps, bitwise_equal_steps=equal,
                bitwise_equal_rate=(equal / steps) if steps else None,
                first_hash_divergence=first)


def load_runs(root: str | Path) -> list[dict]:
    """Load every run file under root, skipping files that are not runs."""
    runs = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("schema", "").startswith("mkparity.run"):
            d["_path"] = str(p)
            runs.append(d)
    return runs


def group_streams(runs: list[dict], key: str = "token_ids") -> dict:
    """{(system, gpu, model, arm, rep): {prompt_id: stream}} for successful runs."""
    g: dict = defaultdict(dict)
    for r in runs:
        if r.get("returncode", 0) != 0 or key not in r:
            continue
        k = (r.get("system"), r.get("gpu", {}).get("name"), r.get("model"), r.get("arm"), r.get("rep", 0))
        g[k][r["prompt_id"]] = r[key]
    return dict(g)


def fmt_rate(c: dict) -> str:
    if c["prompts"] == 0:
        return "n/a"
    lo, hi = c["exact_rate_ci95"]
    return f"{c['exact']}/{c['prompts']} ({100 * c['exact_rate']:.1f}%, CI {100 * lo:.0f} to {100 * hi:.0f}%)"


def markdown_table(rows: list[dict]) -> str:
    head = "| Comparison | Exact outputs | Median first divergence | Survival at 128 / 512 |\n|---|---|---|---|\n"
    lines = []
    for r in rows:
        c = r["result"]
        s = c["survival"]
        surv = " / ".join(f"{100 * s[t]:.0f}%" if t in s else "n/a" for t in (128, 512))
        md = c["median_first_divergence"]
        lines.append(f"| {r['name']} | {fmt_rate(c)} | {md if md is not None else 'none'} | {surv} |")
    return head + "\n".join(lines) + "\n"
