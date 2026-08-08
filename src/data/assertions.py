"""Load-time guardrails for the draft board.

Motivation
----------
In the 2025 draft, ``interactive_draft_assistant.py`` read ``row.get('vorp', 0.0)``
while the CSV column was ``VORP``. Every player loaded with vorp = 0.0, every score
at a position tied, and the tie broke on Python's per-process randomized string
hash. The tool recommended a different (arbitrary) player on every launch and
nothing anywhere reported a problem.

A three-line variance check would have caught it. That is what this module is.

Two severity levels, deliberately distinct:

FATAL   -- the loader is broken. The numbers are not what they claim to be.
           Raises BoardValidationError. Never draft against this.
WARNING -- the loader works but the board is low quality (e.g. it disagrees
           wildly with market consensus). Loud, but not fatal: fixing board
           quality is a projections problem, not a plumbing problem.

The current data/raw/draft_board.csv passes FATAL and fails WARNING, which is
the correct classification -- it loads fine and its projections are bad.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Board-quality thresholds. See module docstring for why these are warnings.
MIN_ADP_SPEARMAN = 0.75      # 2025 board scored -0.25
MIN_TOP200_ADP_COVERAGE = 0.95  # 2025 board scored 0.43 overall
MAX_PLAUSIBLE_PPG = 35.0     # no NFL player averages 35 fantasy ppg


class BoardValidationError(RuntimeError):
    """Raised when a board is structurally broken and must not be drafted against."""


@dataclass
class BoardReport:
    """Outcome of validating a board."""

    n_players: int = 0
    fatal: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing structural is broken (warnings are allowed)."""
        return not self.fatal

    def __str__(self) -> str:
        lines = [f"BoardReport: {self.n_players} players"]
        for key, value in self.stats.items():
            if isinstance(value, float):
                lines.append(f"  {key:<24} {value:.4f}")
            else:
                lines.append(f"  {key:<24} {value}")
        for msg in self.fatal:
            lines.append(f"  FATAL   {msg}")
        for msg in self.warnings:
            lines.append(f"  WARNING {msg}")
        return "\n".join(lines)


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Find a column by case-insensitive match. Returns the real column name.

    This is the exact failure mode that caused the 2025 loss, so resolution is
    explicit and case-insensitive rather than a bare ``df[name]``.
    """
    lookup = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, without requiring scipy."""
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xs) < 3:
        return float("nan")
    rx = pd.Series(xs).rank().to_numpy()
    ry = pd.Series(ys).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def validate_board(
    df: pd.DataFrame,
    *,
    raise_on_fatal: bool = True,
    emit_warnings: bool = True,
) -> BoardReport:
    """Validate a draft-board DataFrame before it is used to make picks.

    Args:
        df: board with at least a player-name, position, and VORP column.
        raise_on_fatal: raise BoardValidationError if a structural check fails.
        emit_warnings: route quality failures through ``warnings.warn``.

    Returns:
        BoardReport describing what passed and what did not.
    """
    report = BoardReport(n_players=len(df))

    if df.empty:
        report.fatal.append("board is empty")
        return _finish(report, raise_on_fatal, emit_warnings)

    vorp_col = _find_column(df, "vorp")
    pos_col = _find_column(df, "position", "pos")
    ppg_col = _find_column(df, "proj_ppg_2026", "proj_ppg_2025", "proj_ppg", "ppg")
    adp_col = _find_column(df, "adp_rank", "adp", "avg")

    # --- FATAL: is the VORP column actually present and actually populated? ---
    if vorp_col is None:
        report.fatal.append(
            f"no VORP column found (columns: {list(df.columns)})"
        )
    else:
        vorp = pd.to_numeric(df[vorp_col], errors="coerce")
        n_valid = int(vorp.notna().sum())
        std = float(vorp.std(skipna=True)) if n_valid > 1 else 0.0
        report.stats["vorp_column"] = vorp_col
        report.stats["vorp_std"] = std
        report.stats["vorp_nonzero_frac"] = (
            float((vorp.fillna(0) != 0).mean()) if n_valid else 0.0
        )

        if n_valid == 0:
            report.fatal.append(f"column '{vorp_col}' has no numeric values")
        elif not math.isfinite(std) or std <= 0.0:
            report.fatal.append(
                f"VORP has zero variance (std={std}) -- every player scores "
                f"identically. This is the 2025 casing bug; check that the "
                f"loader reads '{vorp_col}' and not a differently-cased name."
            )

    if pos_col is None:
        report.fatal.append("no position column found")

    # --- WARNING: is the board plausible? ---
    if ppg_col is not None:
        ppg = pd.to_numeric(df[ppg_col], errors="coerce")
        max_ppg = float(ppg.max(skipna=True)) if ppg.notna().any() else float("nan")
        report.stats["max_proj_ppg"] = max_ppg
        if math.isfinite(max_ppg) and max_ppg >= MAX_PLAUSIBLE_PPG:
            report.warnings.append(
                f"max projected ppg is {max_ppg:.1f} (>= {MAX_PLAUSIBLE_PPG}), "
                f"which is not a real fantasy scoring rate -- check units"
            )

    if adp_col is not None and vorp_col is not None:
        adp = pd.to_numeric(df[adp_col], errors="coerce")
        vorp = pd.to_numeric(df[vorp_col], errors="coerce")

        both = adp.notna() & vorp.notna()
        report.stats["adp_coverage"] = float(adp.notna().mean())

        # Coverage that matters is over the players who actually get drafted.
        top200 = vorp.nlargest(min(200, len(vorp))).index
        cov200 = float(adp.loc[top200].notna().mean()) if len(top200) else 0.0
        report.stats["top200_adp_coverage"] = cov200
        if cov200 < MIN_TOP200_ADP_COVERAGE:
            report.warnings.append(
                f"only {cov200:.0%} of the top-200 board has an ADP "
                f"(want >= {MIN_TOP200_ADP_COVERAGE:.0%}) -- this is usually a "
                f"name-join failure, not missing data"
            )

        # Higher VORP should mean lower (better) ADP, so negate to get a
        # correlation that is positive when board and market agree.
        rho = -_spearman(vorp[both], adp[both])
        report.stats["vorp_adp_spearman"] = rho
        report.stats["n_with_adp"] = int(both.sum())
        if math.isfinite(rho) and rho < MIN_ADP_SPEARMAN:
            report.warnings.append(
                f"board agrees with market ADP at rho={rho:.2f} "
                f"(want >= {MIN_ADP_SPEARMAN}). Below ~0.5 the board is not "
                f"finding edge, it is finding noise."
            )
    elif adp_col is None:
        report.warnings.append("no ADP column -- cannot check against market consensus")

    return _finish(report, raise_on_fatal, emit_warnings)


def validate_players(players: Sequence[Any], *, raise_on_fatal: bool = True) -> BoardReport:
    """Validate loaded Player objects -- catches bugs the DataFrame check cannot.

    ``validate_board`` inspects the CSV; this inspects what the loader actually
    produced. The 2025 bug lived in the gap between the two: the CSV was fine and
    every constructed Player had vorp = 0.0.
    """
    report = BoardReport(n_players=len(players))

    if not players:
        report.fatal.append("no players loaded")
        return _finish(report, raise_on_fatal, True)

    vorps = np.array([float(getattr(p, "vorp", 0.0) or 0.0) for p in players])
    std = float(vorps.std())
    report.stats["vorp_std"] = std
    report.stats["vorp_nonzero_frac"] = float((vorps != 0).mean())
    report.stats["vorp_max"] = float(vorps.max())

    if std <= 0.0:
        report.fatal.append(
            f"all {len(players)} loaded players have identical VORP "
            f"({vorps[0]:.3f}). The board CSV column is probably cased "
            f"differently than the key the loader reads (Series.get is "
            f"case-sensitive)."
        )

    names = [getattr(p, "name", None) for p in players]
    if len(set(names)) != len(names):
        report.warnings.append(
            f"{len(names) - len(set(names))} duplicate player names -- "
            f"Player.__hash__ is hash(name), so duplicates collide in the "
            f"available-players set"
        )

    return _finish(report, raise_on_fatal, True)


def _finish(report: BoardReport, raise_on_fatal: bool, emit_warnings: bool) -> BoardReport:
    if emit_warnings:
        for msg in report.warnings:
            warnings.warn(f"draft board: {msg}", stacklevel=3)
    if report.fatal and raise_on_fatal:
        raise BoardValidationError(
            "draft board failed validation:\n  - " + "\n  - ".join(report.fatal)
        )
    return report
