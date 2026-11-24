"""Named reduction schedules used in the characterization study.

C* schedules emulate behavior found in real megakernels. M* schedules are the
proposed mitigations.
"""
import math

from .emulator import Schedule

SCHEDULES = {
    "C0": Schedule("C0 invariant reference", splitk_down=4, splitk_o=1, combine="ordered_fp32"),
    "C1": Schedule("C1 Hazy-like bf16 store_add", splitk_down=4, splitk_o=1, combine="atomic_bf16",
                   logits_bf16=True),
    "C2": Schedule("C2 fp32 atomicAdd partials", splitk_down=4, splitk_o=2, combine="atomic_fp32"),
    "M1": Schedule("M1 ordered fp32 combine (fix for C1)", splitk_down=4, splitk_o=1,
                   combine="ordered_fp32", logits_bf16=True),
    "C3": Schedule("C3 shape-dependent schedule", splitk_down=lambda R: 4 if R == 1 else 2,
                   splitk_o=lambda R: 2 if R == 1 else 1, combine="ordered_fp32",
                   attn=("count", lambda L: min(math.ceil(L / 32), 8))),
    "M2": Schedule("M2 shape-invariant schedule (fix for C3)", splitk_down=4, splitk_o=2,
                   combine="ordered_fp32", attn=("size", 32)),
}
