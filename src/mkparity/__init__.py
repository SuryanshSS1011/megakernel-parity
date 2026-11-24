"""megakernel-parity: characterizing and mitigating numerical divergence in LLM megakernels."""
from .bits import as_bits, bitwise_equal, first_divergence
from .emulator import Emu, Schedule
from .schedules import SCHEDULES
from .tinyllama import CFG, TinyLlama

__all__ = ["as_bits", "bitwise_equal", "first_divergence", "Emu", "Schedule", "SCHEDULES",
           "CFG", "TinyLlama"]
