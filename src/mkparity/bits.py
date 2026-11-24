"""Bitwise comparison helpers shared by the harnesses and tests."""
import torch

_INT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def as_bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a floating tensor as same-width ints so == is a bitwise test."""
    return t.contiguous().view(_INT[t.element_size()])


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(as_bits(a), as_bits(b)))


def first_divergence(a, b) -> int:
    """Index of the first differing token, len(shorter) on a length mismatch, or -1."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))
