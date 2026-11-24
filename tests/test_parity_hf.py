"""Shape and self-consistency tests for the Hugging Face parity harness, on a
tiny random Qwen3 so no download or GPU is needed."""
import copy
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import parity_hf as P  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    m = transformers.Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    m.config._attn_implementation = "eager"
    return m


@pytest.mark.parametrize("k", [1, 4, 7])
def test_verify_and_decode_shapes(model, k):
    ids = torch.randint(0, 256, (1, 64))
    base = P.prefill(model, ids[:, :48])
    window = ids[:, 48:48 + k + 1]
    ref = P.decode_rows(model, copy.deepcopy(base), window)
    ver = P.verify_rows(model, copy.deepcopy(base), window)
    assert ref.shape == ver.shape == (1, k + 1, 256)
    r = P.compare_logits(ref, ver)
    assert len(r["bitwise_rows"]) == k + 1


def test_reruns_are_bitwise(model):
    ids = torch.randint(0, 256, (1, 56))
    base = P.prefill(model, ids[:, :48])
    w = ids[:, 48:53]
    assert P.bitwise_equal(P.verify_rows(model, copy.deepcopy(base), w),
                           P.verify_rows(model, copy.deepcopy(base), w))
    assert P.bitwise_equal(P.decode_rows(model, copy.deepcopy(base), w),
                           P.decode_rows(model, copy.deepcopy(base), w))
