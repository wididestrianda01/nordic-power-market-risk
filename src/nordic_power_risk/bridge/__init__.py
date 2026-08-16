"""Thesis-bridge analysis (Phase 6)."""

from nordic_power_risk.bridge.mde import (
    FRAMING_INFERENTIAL,
    FRAMING_METHODOLOGY,
    MDE_RELEVANCE_FRACTION,
    MdeGate,
    compute_mde_gate,
    decide_framing,
    minimum_detectable_effect,
    relative_mde,
)
from nordic_power_risk.bridge.run import BridgeResult, PeriodDifficulty, run_thesis_bridge

__all__ = [
    "BridgeResult",
    "FRAMING_INFERENTIAL",
    "FRAMING_METHODOLOGY",
    "MDE_RELEVANCE_FRACTION",
    "MdeGate",
    "PeriodDifficulty",
    "compute_mde_gate",
    "decide_framing",
    "minimum_detectable_effect",
    "relative_mde",
    "run_thesis_bridge",
]
