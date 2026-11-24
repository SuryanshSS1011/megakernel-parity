"""Tiny Llama-style char model (RMSNorm, RoPE, GQA, SwiGLU) for numerics experiments."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(vocab=65, d=256, n_layers=4, n_heads=8, n_kv=2, ffn=1024, max_len=1024, eps=1e-5)


def rope_cache(head_dim, max_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: [T, H, D]; rotate-half convention
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h, kv = c["d"], c["n_heads"], c["n_kv"]
        hd = d // h
        self.ln1 = nn.Parameter(torch.ones(d))
        self.ln2 = nn.Parameter(torch.ones(d))
        self.wq = nn.Linear(d, h * hd, bias=False)
        self.wk = nn.Linear(d, kv * hd, bias=False)
        self.wv = nn.Linear(d, kv * hd, bias=False)
        self.wo = nn.Linear(h * hd, d, bias=False)
        self.wg = nn.Linear(d, c["ffn"], bias=False)
        self.wu = nn.Linear(d, c["ffn"], bias=False)
        self.wd = nn.Linear(c["ffn"], d, bias=False)


class TinyLlama(nn.Module):
    def __init__(self, c=CFG):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["n_layers"])])
        self.lnf = nn.Parameter(torch.ones(c["d"]))
        self.head = nn.Linear(c["d"], c["vocab"], bias=False)
        cos, sin = rope_cache(c["d"] // c["n_heads"], c["max_len"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def rms(x, w, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def forward(self, idx):  # training path, [B, T]
        c = self.c
        B, T = idx.shape
        h, kv = c["n_heads"], c["n_kv"]
        hd = c["d"] // h
        x = self.emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            a = self.rms(x, b.ln1, c["eps"])
            q = b.wq(a).view(B, T, h, hd)
            k = b.wk(a).view(B, T, kv, hd)
            v = b.wv(a).view(B, T, kv, hd)
            q = torch.stack([apply_rope(q[i], cos, sin) for i in range(B)])
            k = torch.stack([apply_rope(k[i], cos, sin) for i in range(B)])
            k = k.repeat_interleave(h // kv, 2)
            v = v.repeat_interleave(h // kv, 2)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2), is_causal=True)
            x = x + b.wo(o.transpose(1, 2).reshape(B, T, h * hd))
            m = self.rms(x, b.ln2, c["eps"])
            x = x + b.wd(F.silu(b.wg(m)) * b.wu(m))
        return self.head(self.rms(x, self.lnf, c["eps"]))
