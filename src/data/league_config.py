"""The single source of truth for league rules.

Three places in this repo hardcode a league config and two of them disagree:

    clean.py:13-28                       half-PPR, Yahoo-style DEF brackets
    src/core/scoring.py:38-48            full PPR (reception 1.0, int -2.0)
    src/utils/data_loader.py:214         a generic 12-team default

Replacement level depends on roster slots, and every VORP on the board depends
on replacement level. Guessing wrong silently rescales the entire board, so this
module refuses to guess: if the league has not been pulled, it raises.

Use::

    from src.data.league_config import load_league_config
    cfg = load_league_config(2025)
    cfg.num_teams, cfg.starting_slots, cfg.flex_slots
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import REPO_ROOT

LEAGUE_ROOT = REPO_ROOT / "data" / "league"

# Yahoo roster slot names -> the position sets that may fill them.
FLEX_SLOT_ELIGIBILITY: Dict[str, frozenset] = {
    "W/R": frozenset({"WR", "RB"}),
    "W/T": frozenset({"WR", "TE"}),
    "R/T": frozenset({"RB", "TE"}),
    "W/R/T": frozenset({"WR", "RB", "TE"}),
    "Q/W/R/T": frozenset({"QB", "WR", "RB", "TE"}),
    "FLEX": frozenset({"WR", "RB", "TE"}),
    "SUPERFLEX": frozenset({"QB", "WR", "RB", "TE"}),
}

# Slots that hold players but never score.
NON_SCORING_SLOTS = frozenset({"BN", "IR", "IL", "NA"})


class LeagueConfigError(RuntimeError):
    """Raised when league rules are missing or unusable."""


@dataclass
class LeagueConfig:
    """Normalized league rules, pulled from the platform rather than assumed."""

    season: int
    league_id: str
    name: Optional[str]
    num_teams: int
    roster_slots: Dict[str, int]
    scoring: Dict[str, Dict[str, Any]]
    playoff_start_week: Optional[int]
    num_playoff_teams: Optional[int]
    start_week: int = 1
    end_week: int = 17
    scoring_type: Optional[str] = None
    source_path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- derived ----------------------------------------------------------

    @property
    def starting_slots(self) -> Dict[str, int]:
        """Slots that actually score, i.e. excluding bench and IR."""
        return {
            slot: n
            for slot, n in self.roster_slots.items()
            if slot not in NON_SCORING_SLOTS and n > 0
        }

    @property
    def bench_size(self) -> int:
        return sum(
            n for slot, n in self.roster_slots.items() if slot in NON_SCORING_SLOTS
        )

    @property
    def roster_size(self) -> int:
        return sum(self.roster_slots.values())

    @property
    def total_rounds(self) -> int:
        """Draft rounds. IR slots are not drafted into, so they don't count."""
        return sum(
            n for slot, n in self.roster_slots.items() if slot not in {"IR", "IL", "NA"}
        )

    @property
    def flex_slots(self) -> Dict[str, frozenset]:
        """Multi-position slots present in this league, slot -> eligible positions."""
        return {
            slot: FLEX_SLOT_ELIGIBILITY[slot]
            for slot in self.starting_slots
            if slot in FLEX_SLOT_ELIGIBILITY
        }

    @property
    def dedicated_slots(self) -> Dict[str, int]:
        """Single-position starting slots, e.g. {'QB': 1, 'RB': 2, ...}."""
        return {
            slot: n
            for slot, n in self.starting_slots.items()
            if slot not in FLEX_SLOT_ELIGIBILITY
        }

    def replacement_rank(self, position: str) -> int:
        """Index of the replacement-level player at ``position``.

        Counts dedicated starters at the position across the league, plus that
        position's share of every flex slot it is eligible for. This is the
        quantity every VORP on the board is measured against.
        """
        starters = self.num_teams * self.dedicated_slots.get(position, 0)
        for slot, eligible in self.flex_slots.items():
            if position in eligible:
                count = self.starting_slots.get(slot, 0)
                starters += round(self.num_teams * count / len(eligible))
        return max(int(starters), 1)

    def scoring_value(self, *patterns: str) -> Optional[float]:
        """Look up a scoring modifier by stat display name (case-insensitive regex).

        Yahoo returns scoring as opaque stat ids, so the pull attaches names and
        this resolves against them::

            cfg.scoring_value(r"^rec(eption)?s?$")   -> 0.5 in a half-PPR league
        """
        for entry in self.scoring.values():
            label = " ".join(
                str(entry.get(k) or "") for k in ("name", "display_name")
            ).strip()
            for pattern in patterns:
                if re.search(pattern, label, flags=re.IGNORECASE):
                    return entry.get("value")
        return None

    @property
    def points_per_reception(self) -> Optional[float]:
        """Settles the half-PPR vs full-PPR conflict from real settings."""
        return self.scoring_value(r"\breception", r"^rec$")

    def summary(self) -> str:
        ppr = self.points_per_reception
        ppr_label = "unknown" if ppr is None else f"{ppr:g}/rec"
        return (
            f"{self.name or 'league'} ({self.season}, id={self.league_id})\n"
            f"  teams        {self.num_teams}\n"
            f"  starters     {self.starting_slots}\n"
            f"  bench        {self.bench_size}\n"
            f"  rounds       {self.total_rounds}\n"
            f"  scoring      {self.scoring_type or '?'} ({ppr_label})\n"
            f"  playoffs     week {self.playoff_start_week}, "
            f"{self.num_playoff_teams} teams"
        )


def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def from_dict(payload: Dict[str, Any], source_path: Optional[Path] = None) -> LeagueConfig:
    """Build a LeagueConfig from a normalized ``league_config.json`` payload."""
    slots = {
        str(k): int(v)
        for k, v in (payload.get("roster_slots") or {}).items()
        if _coerce_int(v) is not None
    }
    if not slots:
        raise LeagueConfigError(
            f"league config has no roster slots ({source_path or 'in-memory'})"
        )

    num_teams = _coerce_int(payload.get("num_teams"))
    if not num_teams:
        raise LeagueConfigError(
            f"league config has no team count ({source_path or 'in-memory'})"
        )

    return LeagueConfig(
        season=_coerce_int(payload.get("season"), 0) or 0,
        league_id=str(payload.get("league_id") or "?"),
        name=payload.get("name"),
        num_teams=num_teams,
        roster_slots=slots,
        scoring=payload.get("scoring") or {},
        playoff_start_week=_coerce_int(payload.get("playoff_start_week")),
        num_playoff_teams=_coerce_int(payload.get("num_playoff_teams")),
        start_week=_coerce_int(payload.get("start_week"), 1) or 1,
        end_week=_coerce_int(payload.get("end_week"), 17) or 17,
        scoring_type=payload.get("scoring_type"),
        source_path=source_path,
        raw=payload.get("raw") or {},
    )


# The shape clean.py assumed. Used ONLY via provisional_config(), which callers
# must ask for by name -- load_league_config() still refuses to guess.
PROVISIONAL_PAYLOAD: Dict[str, Any] = {
    "season": 2026,
    "league_id": "PROVISIONAL",
    "name": "PROVISIONAL (not pulled from Yahoo)",
    "num_teams": 12,
    "scoring_type": "head",
    "start_week": 1,
    "end_week": 17,
    "playoff_start_week": 15,
    "num_playoff_teams": 6,
    "roster_slots": {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1,
        "K": 1, "DEF": 1, "BN": 6,
    },
    "scoring": {
        "pass_yd": {"value": 0.04, "name": "Passing Yards"},
        "pass_td": {"value": 4.0, "name": "Passing Touchdowns"},
        "int": {"value": -1.0, "name": "Interceptions"},
        "rush_yd": {"value": 0.1, "name": "Rushing Yards"},
        "rush_td": {"value": 6.0, "name": "Rushing Touchdowns"},
        "rec_yd": {"value": 0.1, "name": "Receiving Yards"},
        "rec_td": {"value": 6.0, "name": "Receiving Touchdowns"},
        "rec": {"value": 0.5, "name": "Receptions"},
    },
}


def provisional_config() -> LeagueConfig:
    """A stand-in config so downstream work can proceed before the Yahoo pull.

    This is the half-PPR shape ``clean.py`` assumed. It is almost certainly
    close, and it is explicitly NOT trusted: it is reachable only by calling
    this function by name, it is tagged ``league_id="PROVISIONAL"``, and
    :func:`is_provisional` lets pipelines refuse to ship a real board built on
    it. ``load_league_config()`` still raises rather than returning this.
    """
    return from_dict(dict(PROVISIONAL_PAYLOAD))


def is_provisional(config: "LeagueConfig") -> bool:
    """True when a config is the stand-in rather than real pulled settings."""
    return config.league_id == "PROVISIONAL"


def load_or_provisional() -> LeagueConfig:
    """Real settings if pulled, otherwise the stand-in with a loud warning."""
    try:
        return load_league_config()
    except LeagueConfigError:
        import warnings
        warnings.warn(
            "Using PROVISIONAL league settings (12-team half-PPR, 1 flex). "
            "Replacement level and every VORP depend on these. Pull the real "
            "ones before drafting: python -m src.data.yahoo_league --doctor",
            stacklevel=2,
        )
        return provisional_config()


def available_configs() -> List[Path]:
    """Every pulled ``league_config.json``, newest season first."""
    if not LEAGUE_ROOT.is_dir():
        return []
    found = sorted(LEAGUE_ROOT.glob("*/league_config.json"))
    return sorted(found, key=lambda p: p.parent.name, reverse=True)


def load_league_config(
    season: Optional[int] = None, league_id: Optional[str] = None
) -> LeagueConfig:
    """Load league rules pulled from the platform.

    Raises rather than falling back to a default: a wrong roster config silently
    rescales every VORP on the board, which is exactly the class of error that
    should stop the pipeline instead of quietly changing the answer.
    """
    candidates = available_configs()
    if not candidates:
        raise LeagueConfigError(
            "No league config found. Pull it from Yahoo first:\n"
            "    python -m src.data.yahoo_league --discover --season 2025\n"
            "    python -m src.data.yahoo_league --season 2025 --league-id <id>\n"
            f"(searched {LEAGUE_ROOT})"
        )

    if season is not None or league_id is not None:
        filtered = []
        for path in candidates:
            payload = json.loads(path.read_text())
            if season is not None and _coerce_int(payload.get("season")) != season:
                continue
            if league_id is not None and str(payload.get("league_id")) != str(league_id):
                continue
            filtered.append(path)
        if not filtered:
            want = f"season={season} league_id={league_id}"
            have = ", ".join(p.parent.name for p in candidates)
            raise LeagueConfigError(f"no league config matching {want}. have: {have}")
        candidates = filtered

    path = candidates[0]
    return from_dict(json.loads(path.read_text()), source_path=path)


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m src.data.league_config`` -- show the resolved config."""
    try:
        cfg = load_league_config()
    except LeagueConfigError as exc:
        print(exc)
        return 1
    print(cfg.summary())
    print("\nreplacement ranks:")
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        print(f"  {pos:4s} {cfg.replacement_rank(pos)}")
    print(f"\nsource: {cfg.source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
