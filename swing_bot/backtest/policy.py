"""Small policy helpers for episode backtests.

This module intentionally contains no model logic.  It only turns already-built
OOF entry/exit predictions into deterministic state transitions for valid-fold
research.  Test evaluation should use a locked config in a separate script.
"""
from __future__ import annotations

from dataclasses import dataclass


class PolicyConfigError(ValueError):
    """Raised when an episode policy config is inconsistent."""


@dataclass(frozen=True)
class EpisodePolicyConfig:
    """Configuration for the first episode-backtest policy.

    ``entry_threshold_bps`` is applied to the entry model's predicted net return.
    ``hold_threshold_bps`` is applied to the exit model's predicted hold-delta.
    Roundtrip cost is charged exactly once per completed episode.
    """

    entry_threshold_bps: float = 20.0
    hold_threshold_bps: float = 0.0
    roundtrip_cost_bps: float = 15.0
    max_hold_minutes: int = 240
    decision_interval_minutes: int = 5
    exit_confirm_bars: int = 1
    cooldown_after_exit_minutes: int = 15
    same_side_reentry_block_minutes: int = 30
    hard_stop_bps: float | None = None
    max_round_trips_per_day: int | None = None

    def __post_init__(self) -> None:
        if self.max_hold_minutes <= 0:
            raise PolicyConfigError("max_hold_minutes must be positive")
        if self.decision_interval_minutes <= 0:
            raise PolicyConfigError("decision_interval_minutes must be positive")
        if self.exit_confirm_bars <= 0:
            raise PolicyConfigError("exit_confirm_bars must be positive")
        if self.cooldown_after_exit_minutes < 0:
            raise PolicyConfigError("cooldown_after_exit_minutes must be non-negative")
        if self.same_side_reentry_block_minutes < 0:
            raise PolicyConfigError("same_side_reentry_block_minutes must be non-negative")
        if self.roundtrip_cost_bps < 0:
            raise PolicyConfigError("roundtrip_cost_bps must be non-negative")
        if self.max_round_trips_per_day is not None and self.max_round_trips_per_day <= 0:
            raise PolicyConfigError("max_round_trips_per_day must be positive when provided")


def should_enter(pred_entry_net_bps: float, config: EpisodePolicyConfig) -> bool:
    """Return True when a flat-state entry signal passes the threshold."""
    try:
        return float(pred_entry_net_bps) > float(config.entry_threshold_bps)
    except Exception:
        return False


def should_hold(pred_exit_hold_delta_bps: float, config: EpisodePolicyConfig) -> bool:
    """Return True when a position should remain open for this decision row."""
    try:
        return float(pred_exit_hold_delta_bps) > float(config.hold_threshold_bps)
    except Exception:
        return False
