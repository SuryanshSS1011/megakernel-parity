"""Megakernel numerics emulator.

Runs the trained TinyLlama with explicit, configurable reduction schedules that
mirror what real megakernels do. Every row is computed with identical tensor
shapes, so the base arithmetic is row-invariant by construction. Any verify vs
decode or run-to-run difference therefore comes only from the schedule policy
under test.

Storage is bf16 between ops (as in Hazy and MPK). Accumulation is fp32.

Policies (per schedule):
  splitk_down / splitk_o : number of K-partials for down_proj / o_proj, either an
                           int or a callable of R (rows in this forward)
  combine                : "ordered_fp32" | "atomic_bf16" | "atomic_fp32"
                           atomic_* add partials into the residual in a random
                           arrival order (emulating TMA store_add / atomicAdd)
  attn                   : ("single",) | ("count", fn(len)->P) | ("size", Z)
  logits_bf16            : round logits to bf16 (Hazy) or keep fp32
"""
import math
from dataclasses import dataclass
from typing import Callable, Union

import torch

from .tinyllama import TinyLlama

BF = torch.bfloat16
CHUNK = 64  # attention KV tile


def bf(x):
    return x.to(BF)


@dataclass
class Schedule:
    name: str
    splitk_down: Union[int, Callable] = 1
    splitk_o: Union[int, Callable] = 1
    combine: str = "ordered_fp32"
    attn: tuple = ("single",)
    logits_bf16: bool = False

    def sk(self, v, R):
        return v(R) if callable(v) else v


class Emu:
    @classmethod
    def from_checkpoint(cls, path):
        d = torch.load(path, weights_only=False)
        m = TinyLlama(d["cfg"])
        m.load_state_dict(d["state"])
        return cls(m, d["chars"])

    def __init__(self, model: TinyLlama, chars=None):
        m = model.eval()
        self.cfg = c = m.c
        self.chars = chars if chars is not None else [chr(32 + i) for i in range(c["vocab"])]
        self.ref = m
        self.H, self.KV, self.hd = c["n_heads"], c["n_kv"], c["d"] // c["n_heads"]
        self.L = c["n_layers"]
        self.eps = c["eps"]
        W = lambda t: bf(t.detach())
        self.emb = W(m.emb.weight)
        self.blocks = []
        for b in m.blocks:
            self.blocks.append(dict(
                ln1=b.ln1.detach().float(), ln2=b.ln2.detach().float(),
                wqkv=W(torch.cat([b.wq.weight, b.wk.weight, b.wv.weight], 0)),
                wo=W(b.wo.weight), wgu=W(torch.cat([b.wg.weight, b.wu.weight], 0)),
                wd=W(b.wd.weight)))
        self.lnf = m.lnf.detach().float()
        self.head = W(m.head.weight)
        self.cos, self.sin = m.cos.float(), m.sin.float()

    # ---------- primitives (row-invariant by construction) ----------
    def rms(self, x, w):  # x bf16 [R, d] -> bf16
        xf = x.float()
        out = []
        for r in range(xf.shape[0]):
            v = xf[r]
            out.append(v * torch.rsqrt((v * v).mean() + self.eps) * w)
        return bf(torch.stack(out))

    @staticmethod
    def partials(x, W, S):
        """x bf16 [R,K], W bf16 [N,K] -> list of S fp32 partials, each [R,N].
        Each row is a separate gemv with identical shape, so row results do not
        depend on R."""
        K = W.shape[1]
        assert K % S == 0
        ks = K // S
        Wf = W.float()
        parts = []
        for s in range(S):
            sl = slice(s * ks, (s + 1) * ks)
            parts.append(torch.stack([Wf[:, sl] @ x[r, sl].float() for r in range(x.shape[0])]))
        return parts

    def matvec(self, x, W, S=1):  # plain projection, ordered fp32, one bf16 round
        p = self.partials(x, W, S)
        acc = p[0].clone()
        for q in p[1:]:
            acc = acc + q
        return bf(acc)

    def matvec_residual(self, x, W, res, S, combine, rng):
        p = self.partials(x, W, S)
        order = list(range(S))
        if combine != "ordered_fp32" and rng is not None:
            rng.shuffle(order)
        if combine == "ordered_fp32":
            acc = p[0].clone()
            for q in p[1:]:
                acc = acc + q
            return bf(res.float() + acc)
        if combine == "atomic_bf16":  # TMA store_add of bf16 partials into bf16 residual
            out = res.clone()
            for s in order:
                out = bf(out.float() + bf(p[s]).float())
            return out
        if combine == "atomic_fp32":  # fp32 atomicAdd into an fp32 accumulator
            acc = res.float().clone()
            for s in order:
                acc = acc + p[s]
            return bf(acc)
        raise ValueError(combine)

    def rope(self, x, pos):  # x fp32 [nh, hd]
        d = self.hd // 2
        c, s = self.cos[pos], self.sin[pos]
        x1, x2 = x[:, :d], x[:, d:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

    def attn_partial(self, q, K, V, lo, hi):
        """q fp32 [H,hd]; K,V bf16 [Lk,KV,hd]; online softmax over CHUNK tiles
        aligned at lo. Returns (m [H], l [H], o [H,hd]) in fp32."""
        g = self.H // self.KV
        m = torch.full((self.H,), -float("inf"))
        l = torch.zeros(self.H)
        o = torch.zeros(self.H, self.hd)
        scale = 1.0 / math.sqrt(self.hd)
        for c0 in range(lo, hi, CHUNK):
            c1 = min(c0 + CHUNK, hi)
            k = K[c0:c1].float().repeat_interleave(g, 1)  # [n,H,hd]
            v = V[c0:c1].float().repeat_interleave(g, 1)
            s = torch.einsum("hd,nhd->hn", q, k) * scale
            mn = torch.maximum(m, s.max(-1).values)
            a = torch.exp(m - mn)
            p = torch.exp(s - mn[:, None])
            l = l * a + p.sum(-1)
            o = o * a[:, None] + torch.einsum("hn,nhd->hd", p, v)
            m = mn
        return m, l, o

    def attention(self, q, K, V, length, policy, P_override=None):
        kind = policy[0]
        if kind == "single":
            bounds = [(0, length)]
        elif kind == "size":
            Z = policy[1]
            bounds = [(a, min(a + Z, length)) for a in range(0, length, Z)]
        elif kind == "count":
            P = P_override if P_override is not None else policy[1](length)
            P = max(1, min(P, length))
            step = math.ceil(length / P)
            bounds = [(a, min(a + step, length)) for a in range(0, length, step)]
        else:
            raise ValueError(policy)
        parts = [self.attn_partial(q, K, V, a, b) for a, b in bounds]
        if len(parts) == 1:
            m, l, o = parts[0]
            return o / l[:, None]
        M = torch.stack([p[0] for p in parts]).max(0).values
        num = torch.zeros(self.H, self.hd)
        den = torch.zeros(self.H)
        for m, l, o in parts:  # ordered merge (Hazy attention_reduction is ordered)
            w = torch.exp(m - M)
            num = num + o * w[:, None]
            den = den + l * w
        return num / den[:, None]

    # ---------- forward over R rows ----------
    def forward_rows(self, toks, pos0, cache, sch: Schedule, rng=None):
        """toks: list of R token ids at positions pos0..pos0+R-1. cache: list per
        layer of [K bf16 [T,KV,hd], V bf16] (mutated: rows appended). Returns
        logits [R, V] fp32."""
        R = len(toks)
        x = self.emb[torch.tensor(toks)]  # bf16 [R,d]
        H, KV, hd = self.H, self.KV, self.hd
        for li, b in enumerate(self.blocks):
            a = self.rms(x, b["ln1"])
            qkv = self.matvec(a, b["wqkv"]).float()
            q = qkv[:, : H * hd].view(R, H, hd)
            k = qkv[:, H * hd: H * hd + KV * hd].view(R, KV, hd)
            v = qkv[:, H * hd + KV * hd:].view(R, KV, hd)
            Kc, Vc = cache[li]
            newK, newV, qs = [], [], []
            for r in range(R):
                p = pos0 + r
                qs.append(bf(self.rope(q[r], p)).float())
                newK.append(bf(self.rope(k[r], p)))
                newV.append(bf(v[r]))
            Kc = torch.cat([Kc, torch.stack(newK)], 0)
            Vc = torch.cat([Vc, torch.stack(newV)], 0)
            cache[li] = [Kc, Vc]
            # In a fused verify, a count-based split is chosen once from the
            # longest row; in decode it is chosen per step.
            P_shared = None
            if sch.attn[0] == "count" and R > 1:
                P_shared = sch.attn[1](pos0 + R)
            att = []
            for r in range(R):
                length = pos0 + r + 1
                att.append(bf(self.attention(qs[r], Kc, Vc, length, sch.attn, P_shared).reshape(-1)))
            att = torch.stack(att)
            x = self.matvec_residual(att, b["wo"], x, sch.sk(sch.splitk_o, R), sch.combine, rng)
            mm = self.rms(x, b["ln2"])
            gu = self.matvec(mm, b["wgu"]).float()
            f = gu.shape[1] // 2
            hmid = bf(torch.nn.functional.silu(gu[:, :f]) * gu[:, f:])
            x = self.matvec_residual(hmid, b["wd"], x, sch.sk(sch.splitk_down, R), sch.combine, rng)
        z = self.rms(x, self.lnf)
        logits = torch.stack([self.head.float() @ z[r].float() for r in range(R)])
        return bf(logits).float() if sch.logits_bf16 else logits

    def empty_cache(self):
        return [[torch.zeros(0, self.KV, self.hd, dtype=BF), torch.zeros(0, self.KV, self.hd, dtype=BF)]
                for _ in range(self.L)]

    def prefill(self, toks, sch, rng=None):
        """Token-by-token prefill (M=1), as the latency megakernels do."""
        cache = self.empty_cache()
        logits = None
        for i, t in enumerate(toks):
            logits = self.forward_rows([t], i, cache, sch, rng)
        return cache, logits[-1]

    def encode(self, s):
        idx = {c: i for i, c in enumerate(self.chars)}
        return [idx[c] for c in s]

    def decode_str(self, ids):
        return "".join(self.chars[i] for i in ids)
