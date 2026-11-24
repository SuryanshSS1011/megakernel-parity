"""X6: longer, larger runs for schedules whose perturbations were too small to
flip greedy tokens in the short runs. C3 spec-vs-greedy and C2 run-to-run, with
40 prompts x 384 tokens, plus per-row flip analysis against top-2 margins."""
import argparse
import json
import random
import time

import torch

from mkparity import SCHEDULES, Emu

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="data/ckpt.pt")
ap.add_argument("--data", default="data/tinyshakespeare.txt")
ap.add_argument("--prompts", type=int, default=40)
ap.add_argument("--gen", type=int, default=384)
ap.add_argument("--out", default="results/emulator/long_runs.json")
args = ap.parse_args()

torch.set_num_threads(1)
NP, GEN, PLEN = args.prompts, args.gen, 48
emu = Emu.from_checkpoint(args.ckpt)
C2, C3 = SCHEDULES["C2"], SCHEDULES["C3"]
text = open(args.data).read()
val = text[int(0.9 * len(text)):]
prng = random.Random(99)
prompts = []
while len(prompts) < NP:
    i = prng.randrange(0, len(val) - PLEN)
    prompts.append(emu.encode(val[i:i + PLEN]))


def greedy(prompt, sch, seed):
    rng = random.Random(seed) if seed is not None else None
    cache, logits = emu.prefill(prompt, sch, rng)
    out, pos = [], len(prompt)
    for _ in range(GEN):
        t = int(logits.argmax()); out.append(t)
        logits = emu.forward_rows([t], pos, cache, sch, rng)[-1]; pos += 1
    return out


def prompt_lookup(ctx, k, nmax=3, nmin=2):
    for n in range(nmax, nmin - 1, -1):
        tail = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == tail:
                return ctx[i + n:i + n + k]
    return []


def spec(prompt, sch, k):
    cache, _ = emu.prefill(prompt[:-1], sch, None)
    ctx, out = list(prompt), []
    while len(out) < GEN:
        draft = prompt_lookup(ctx, k)
        rows = [ctx[-1]] + draft
        pos0 = len(ctx) - 1
        lg = emu.forward_rows(rows, pos0, cache, sch, None)
        acc, new = 0, []
        for j in range(len(rows)):
            t = int(lg[j].argmax()); new.append(t)
            if j < len(draft) and draft[j] == t:
                acc += 1; continue
            break
        keep = pos0 + 1 + acc
        cache[:] = [[K[:keep], V[:keep]] for K, V in cache]
        ctx += new; out += new
    return out[:GEN]


def fd(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1)


t0 = time.time()
res = {}
base = [greedy(p, C3, None) for p in prompts]
sp = [spec(p, C3, 4) for p in prompts]
f = [fd(a, b) for a, b in zip(base, sp)]
res["C3_spec_vs_greedy"] = dict(prompts=NP, gen=GEN, exact=sum(x == -1 for x in f),
                                first_divergences=sorted(x for x in f if x >= 0))
print(res, time.time() - t0, flush=True)
a = [greedy(p, C2, 0) for p in prompts]
b = [greedy(p, C2, 1) for p in prompts]
f = [fd(x, y) for x, y in zip(a, b)]
res["C2_seed0_vs_seed1"] = dict(prompts=NP, gen=GEN, exact=sum(x == -1 for x in f),
                                first_divergences=sorted(x for x in f if x >= 0))
print(res, time.time() - t0, flush=True)
json.dump(res, open(args.out, "w"), indent=1)
