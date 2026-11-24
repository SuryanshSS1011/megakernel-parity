"""Tests for the bitwise helpers and the vLLM compare step (no GPU needed)."""
import json
import sys
from pathlib import Path

import torch

from mkparity import as_bits, bitwise_equal, first_divergence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import parity_vllm  # noqa: E402


def test_bitwise_equal_sees_one_ulp():
    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    b = a.clone()
    b.view(torch.int16)[0] += 1
    assert bitwise_equal(a, a.clone())
    assert not bitwise_equal(a, b)


def test_bitwise_equal_distinguishes_signed_zero():
    assert not bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_as_bits_widths():
    for dt, it in [(torch.float16, torch.int16), (torch.bfloat16, torch.int16),
                   (torch.float32, torch.int32), (torch.float64, torch.int64)]:
        assert as_bits(torch.zeros(3, dtype=dt)).dtype == it


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) == -1
    assert first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_vllm_compare(tmp_path):
    meta = {"arm": "x"}
    a = {"meta": meta, "records": [
        {"tokens": [1, 2, 3], "logprob_hex": [(-0.5).hex(), (-0.25).hex(), (-1.0).hex()]},
        {"tokens": [4, 5], "logprob_hex": [(-0.1).hex(), (-0.2).hex()]}]}
    b = {"meta": meta, "records": [
        {"tokens": [1, 2, 3], "logprob_hex": [(-0.5).hex(), (-0.2500001).hex(), (-1.0).hex()]},
        {"tokens": [4, 6], "logprob_hex": [(-0.1).hex(), (-0.2).hex()]}]}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    r = parity_vllm.compare(str(pa), str(pb))
    assert r["token_exact"] == 1
    assert r["logprob_bitwise"] == 0
    assert r["first_token_divergence"] == [-1, 1]
    assert r["first_logprob_divergence"] == [1, -1]
