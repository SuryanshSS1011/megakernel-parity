"""Characterization experiments for megakernel reduction schedules (CPU emulation).

X1 run-to-run determinism under atomic-style partial accumulation
X2 verify(M=k+1) vs decode(M=1) parity under shape-dependent schedules
X3 end-to-end speculative decoding (prompt lookup) vs plain greedy decode
X4 fidelity vs the fp32 reference model (KL, top-1 agreement)
X5 logit-margin distribution vs observed perturbation sizes
"""
import argparse
import json
import random
import time

import torch

from mkparity import SCHEDULES, Emu

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="data/ckpt.pt")
ap.add_argument("--data", default="data/tinyshakespeare.txt")
ap.add_argument("--prompts", type=int, default=16)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--out", default="results/emulator/characterization.json")
args = ap.parse_args()

torch.set_num_threads(1)
N_PROMPTS, GEN, PLEN, SEEDS = args.prompts, args.gen, 48, args.seeds

S = SCHEDULES

emu = Emu.from_checkpoint(args.ckpt)
text = open(args.data).read()
val = text[int(0.9 * len(text)):]
prng = random.Random(1234)
prompts = []
while len(prompts) < N_PROMPTS:
    i = prng.randrange(0, len(val) - PLEN)
    prompts.append(emu.encode(val[i:i + PLEN]))


def bits_equal(a, b):
    return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def greedy(prompt, sch, seed):
    rng = random.Random(seed) if seed is not None else None
    cache, logits = emu.prefill(prompt, sch, rng)
    out, pos = [], len(prompt)
    for _ in range(GEN):
        t = int(logits.argmax())
        out.append(t)
        logits = emu.forward_rows([t], pos, cache, sch, rng)[-1]
        pos += 1
    return out


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def decode_logits_and_cache(seq, sch, seed=None):
    rng = random.Random(seed) if seed is not None else None
    cache = emu.empty_cache()
    rows = []
    for i, t in enumerate(seq):
        rows.append(emu.forward_rows([t], i, cache, sch, rng)[-1])
    return torch.stack(rows), cache


def verify_rows(seq, cache_full, p, k, sch):
    cache = [[K[:p].clone(), V[:p].clone()] for K, V in cache_full]
    return emu.forward_rows(seq[p:p + k + 1], p, cache, sch, None)


def prompt_lookup(ctx, k, nmax=3, nmin=2):
    for n in range(nmax, nmin - 1, -1):
        if len(ctx) <= n:
            continue
        tail = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == tail:
                return ctx[i + n:i + n + k]
    return []


def spec_decode(prompt, sch, k, seed):
    rng = random.Random(seed) if seed is not None else None
    cache, logits = emu.prefill(prompt[:-1], sch, rng) if len(prompt) > 1 else (emu.empty_cache(), None)
    ctx = list(prompt)
    out = []
    steps = accepted_total = proposed_total = 0
    while len(out) < GEN:
        draft = prompt_lookup(ctx, k)
        rows = [ctx[-1]] + draft
        pos0 = len(ctx) - 1
        lg = emu.forward_rows(rows, pos0, cache, sch, rng)
        steps += 1
        acc = 0
        new = []
        for j in range(len(rows)):
            t = int(lg[j].argmax())
            new.append(t)
            if j < len(draft) and draft[j] == t:
                acc += 1
                continue
            break
        accepted_total += acc
        proposed_total += len(draft)
        # keep KV for the rows whose inputs are committed: ctx[-1] + accepted drafts
        keep = pos0 + 1 + acc
        cache[:] = [[K[:keep], V[:keep]] for K, V in cache]
        ctx += new
        out += new
    return out[:GEN], dict(steps=steps, accepted=accepted_total, proposed=proposed_total)


results = {}
t0 = time.time()

# ---------------- X1: run-to-run determinism ----------------
x1 = {}
for key in ["C0", "C1", "C2", "M1"]:
    sch = S[key]
    seeds = list(range(SEEDS)) if sch.combine != "ordered_fp32" else [0, 1]
    outs = {s: [greedy(p, sch, s) for p in prompts] for s in seeds}
    identical = sum(all(outs[s][i] == outs[seeds[0]][i] for s in seeds) for i in range(len(prompts)))
    fds = [first_div(outs[seeds[0]][i], outs[s][i]) for s in seeds[1:] for i in range(len(prompts))]
    div = [f for f in fds if f >= 0]
    x1[key] = dict(name=sch.name, seeds=len(seeds), prompts=len(prompts),
                   prompts_identical_across_seeds=identical,
                   pairwise_divergent_fraction=len(div) / len(fds),
                   median_first_divergence=(sorted(div)[len(div) // 2] if div else None),
                   min_first_divergence=(min(div) if div else None))
    print("X1", key, x1[key], f"t={time.time()-t0:.0f}s", flush=True)
results["X1_run_to_run"] = x1

# Reference sequences: prompt + C0 greedy continuation (fixed token streams)
ref_seqs = [p + greedy(p, S["C0"], None) for p in prompts]

# Logit-level run-to-run (teacher forced): seed 0 vs seed 1
x1b = {}
for key in ["C1", "C2", "M1"]:
    eq = tot = 0
    mx = 0.0
    for seq in ref_seqs:
        a, _ = decode_logits_and_cache(seq, S[key], 0)
        b, _ = decode_logits_and_cache(seq, S[key], 1)
        for r in range(PLEN - 1, len(seq)):
            tot += 1
            eq += bits_equal(a[r], b[r])
            mx = max(mx, float((a[r] - b[r]).abs().max()))
    x1b[key] = dict(rows=tot, bitwise_equal_rate=eq / tot, max_abs_logit_diff=mx)
    print("X1b", key, x1b[key], flush=True)
results["X1b_logits_run_to_run"] = x1b

# ---------------- X2: verify vs decode parity ----------------
x2 = {}
for key in ["C0", "C3", "M2", "C1"]:
    sch = S[key]
    for k in [1, 3, 7]:
        eq = flips = tot = 0
        mx = 0.0
        for seq in ref_seqs:
            dl, cache = decode_logits_and_cache(seq, sch, None)
            for p in range(PLEN, len(seq) - k - 1, k + 1):
                vr = verify_rows(seq, cache, p, k, sch)
                for j in range(k + 1):
                    tot += 1
                    d = dl[p + j]
                    eq += bits_equal(d, vr[j])
                    flips += int(d.argmax() != vr[j].argmax())
                    mx = max(mx, float((d - vr[j]).abs().max()))
        x2[f"{key}_k{k}"] = dict(name=sch.name, k=k, M=k + 1, rows=tot, bitwise_equal_rate=eq / max(1, tot),
                                 argmax_flip_rate=flips / max(1, tot), max_abs_logit_diff=mx)
        print("X2", key, k, x2[f"{key}_k{k}"], flush=True)
results["X2_verify_vs_decode"] = x2

# ---------------- X3: end-to-end spec decoding ----------------
x3 = {}
for key in ["C0", "C3", "M2", "C1"]:
    sch = S[key]
    k = 4
    seed = 0 if sch.combine != "ordered_fp32" else None
    base = [greedy(p, sch, seed) for p in prompts]
    match = 0
    fds = []
    st = dict(steps=0, accepted=0, proposed=0)
    for i, p in enumerate(prompts):
        o, s = spec_decode(p, sch, k, seed)
        for kk in st:
            st[kk] += s[kk]
        f = first_div(base[i], o)
        match += f == -1
        if f >= 0:
            fds.append(f)
    x3[key] = dict(name=sch.name, k=k, prompts=len(prompts), exact_match=match,
                   exact_match_rate=match / len(prompts),
                   median_first_divergence=(sorted(fds)[len(fds) // 2] if fds else None),
                   acceptance_rate=st["accepted"] / max(1, st["proposed"]),
                   tokens_per_step=len(prompts) * GEN / st["steps"])
    print("X3", key, x3[key], flush=True)
results["X3_spec_vs_greedy"] = x3

# ---------------- X4: fidelity vs fp32 reference ----------------
x4 = {}
with torch.no_grad():
    for key in ["C0", "C1", "M1", "C2"]:
        kl = agree = tot = 0.0
        for seq in ref_seqs:
            ref = emu.ref(torch.tensor([seq])).float()[0]
            a, _ = decode_logits_and_cache(seq, S[key], 0 if S[key].combine != "ordered_fp32" else None)
            lp_r = torch.log_softmax(ref, -1)
            lp_a = torch.log_softmax(a, -1)
            for r in range(PLEN - 1, len(seq)):
                kl += float((lp_r[r].exp() * (lp_r[r] - lp_a[r])).sum())
                agree += int(ref[r].argmax() == a[r].argmax())
                tot += 1
        x4[key] = dict(name=S[key].name, rows=int(tot), mean_kl_vs_fp32=kl / tot, top1_agreement=agree / tot)
        print("X4", key, x4[key], flush=True)
results["X4_fidelity"] = x4

# ---------------- X5: margins ----------------
margins = []
for seq in ref_seqs:
    a, _ = decode_logits_and_cache(seq, S["C0"], None)
    for r in range(PLEN - 1, len(seq)):
        t2 = a[r].topk(2).values
        margins.append(float(t2[0] - t2[1]))
margins.sort()
q = lambda f: margins[int(f * (len(margins) - 1))]
results["X5_margins"] = dict(n=len(margins), p01=q(0.01), p05=q(0.05), p10=q(0.10), p50=q(0.5),
                             frac_below_0p01=sum(m < 0.01 for m in margins) / len(margins),
                             frac_below_0p05=sum(m < 0.05 for m in margins) / len(margins),
                             frac_below_0p1=sum(m < 0.1 for m in margins) / len(margins))
print("X5", results["X5_margins"], flush=True)
results["meta"] = dict(prompts=N_PROMPTS, gen=GEN, prompt_len=PLEN, seeds=SEEDS,
                       model=emu.cfg, runtime_s=time.time() - t0)
json.dump(results, open(args.out, "w"), indent=1)
print("done", time.time() - t0)
