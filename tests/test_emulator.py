"""CPU tests for the reduction-schedule emulator.

These guard the properties the characterization depends on. The reference and
mitigation schedules must be bitwise row-invariant and run-to-run
deterministic, and the hazard schedules must actually produce the hazard.
"""
import copy
import random

import pytest
import torch

from mkparity import SCHEDULES, Emu, TinyLlama, bitwise_equal

SMALL = dict(vocab=48, d=64, n_layers=2, n_heads=4, n_kv=2, ffn=256, max_len=256, eps=1e-5)


@pytest.fixture(scope="module")
def emu():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    return Emu(TinyLlama(dict(SMALL)))


@pytest.fixture(scope="module")
def seq():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, SMALL["vocab"], (72,), generator=g).tolist()


def decode_all(emu, seq, sch, seed=None):
    rng = random.Random(seed) if seed is not None else None
    cache = emu.empty_cache()
    rows = [emu.forward_rows([t], i, cache, sch, rng)[-1] for i, t in enumerate(seq)]
    return torch.stack(rows), cache


def verify(emu, seq, cache, p, k, sch):
    c = [[K[:p].clone(), V[:p].clone()] for K, V in cache]
    return emu.forward_rows(seq[p:p + k + 1], p, c, sch, None)


def test_reference_matches_fp32_model(emu, seq):
    logits, _ = decode_all(emu, seq, SCHEDULES["C0"])
    with torch.no_grad():
        ref = emu.ref(torch.tensor([seq]))[0].float()
    assert float((logits - ref).abs().max()) < 0.1
    assert float((logits.argmax(-1) == ref.argmax(-1)).float().mean()) > 0.95


@pytest.mark.parametrize("key", ["C0", "M1", "M2", "C3"])
def test_ordered_schedules_are_deterministic(emu, seq, key):
    a, _ = decode_all(emu, seq, SCHEDULES[key])
    b, _ = decode_all(emu, seq, SCHEDULES[key])
    assert bitwise_equal(a, b)


@pytest.mark.parametrize("key", ["C0", "M2"])
@pytest.mark.parametrize("k", [1, 3, 7])
def test_invariant_schedules_verify_equals_decode(emu, seq, key, k):
    dl, cache = decode_all(emu, seq, SCHEDULES[key])
    for p in (40, 55):
        vr = verify(emu, seq, cache, p, k, SCHEDULES[key])
        assert bitwise_equal(vr, dl[p:p + k + 1]), f"p={p}"


def test_splitk_count_changes_results(emu):
    # A random untrained model is too small-valued for C3 to show at the logit
    # level, so the hazard is checked at the op level. The trained-model
    # result lives in results/emulator/characterization.json.
    torch.manual_seed(3)
    x = torch.randn(8, 1024).bfloat16()
    W = (torch.randn(4096, 1024) * 0.05).bfloat16()
    res = torch.zeros(8, 4096).bfloat16()
    a = emu.matvec_residual(x, W, res, 4, "ordered_fp32", None)
    b = emu.matvec_residual(x, W, res, 2, "ordered_fp32", None)
    assert not bitwise_equal(a, b)
    assert bitwise_equal(a, emu.matvec_residual(x, W, res, 4, "ordered_fp32", None))


def test_attention_split_count_changes_results(emu):
    torch.manual_seed(4)
    L, KV, hd = 200, emu.KV, emu.hd
    K = (torch.randn(L, KV, hd) * 2).bfloat16()
    V = torch.randn(L, KV, hd).bfloat16()
    q = torch.randn(emu.H, hd) * 2
    one = emu.attention(q, K, V, L, ("count", lambda n: 1))
    three = emu.attention(q, K, V, L, ("count", lambda n: 3))
    assert not torch.equal(one, three)
    fixed_a = emu.attention(q, K, V, L, ("size", 32))
    fixed_b = emu.attention(q, K, V, L, ("size", 32))
    assert torch.equal(fixed_a, fixed_b)


def test_bf16_atomics_are_nondeterministic(emu, seq):
    a, _ = decode_all(emu, seq, SCHEDULES["C1"], seed=0)
    b, _ = decode_all(emu, seq, SCHEDULES["C1"], seed=1)
    assert not bitwise_equal(a, b)


def test_ordered_combine_fixes_atomics(emu, seq):
    a, _ = decode_all(emu, seq, SCHEDULES["M1"], seed=0)
    b, _ = decode_all(emu, seq, SCHEDULES["M1"], seed=1)
    assert bitwise_equal(a, b)


def test_cache_is_not_aliased(emu, seq):
    sch = SCHEDULES["C0"]
    dl, cache = decode_all(emu, seq, sch)
    before = copy.deepcopy(cache)
    verify(emu, seq, cache, 40, 3, sch)
    assert all(bitwise_equal(K, K0) and bitwise_equal(V, V0)
               for (K, V), (K0, V0) in zip(cache, before))
