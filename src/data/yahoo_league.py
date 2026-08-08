"""Pull real league configuration and history from the Yahoo Fantasy Sports API.

Why this exists
---------------
Three places in this repo hardcode a league config, and they disagree:
``clean.py`` says half-PPR with Yahoo-style DEF brackets, ``src/core/scoring.py``
says full PPR. Nothing reconciles them. Replacement level -- and therefore every
VORP on the board -- depends on which one is real.

This module makes the league itself the source of truth.

What it pulls (2025, for calibration and backtesting)
    settings.json        exact scoring modifiers, roster slots, playoff format
    league_config.json   the above, normalized to what the rest of the code wants
    draft_picks.csv      every pick by every manager  <- opponent calibration
    standings.csv        final standings
    managers.csv         team id -> manager
    weekly_matchups.csv  per-week scores and H2H results
    weekly_rosters.csv   who was actually STARTED each week  (--rosters, slow)

Authentication
--------------
Yahoo uses 3-legged OAuth2. You need a consumer key/secret from
https://developer.yahoo.com/apps/ (create an app, "Fantasy Sports" read
permission, redirect URI ``oob``).

Run the consent flow ONCE, locally::

    python -m src.data.yahoo_league --auth

That writes tokens to the cache .env (see src/data/paths.py). To use the same
session from Colab, copy that .env into your Drive cache dir -- yfpy wants a
localhost redirect, which is not a fight worth having inside a notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .paths import cache_dir, ensure, env_file, is_colab, league_dir

# Yahoo's NFL "game code". Game *keys* are per-season and resolved via the API.
GAME_CODE = "nfl"


class YahooAuthError(RuntimeError):
    """Raised when credentials are missing or the OAuth handshake fails."""


# ---------------------------------------------------------------------------
# yfpy model -> plain python
# ---------------------------------------------------------------------------

def _as_dict(obj: Any) -> Any:
    """Recursively convert a yfpy model into plain dicts/lists/scalars.

    yfpy models carry their payload on ``_extracted_data`` and nest other models
    inside it, so a plain ``vars()`` leaves model objects in the tree and json
    serialization fails.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    data = getattr(obj, "_extracted_data", None)
    if isinstance(data, dict):
        return {k: _as_dict(v) for k, v in data.items()}
    if hasattr(obj, "__dict__"):
        return {
            k: _as_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """First present attribute/key among ``names``."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class LeagueRef:
    """Enough to identify one season of one league."""

    league_id: str
    name: str
    season: int
    num_teams: Optional[int] = None
    league_key: Optional[str] = None
    draft_status: Optional[str] = None

    def __str__(self) -> str:
        teams = f"{self.num_teams} teams" if self.num_teams else "? teams"
        return (
            f"{self.name!r}  id={self.league_id}  {self.season}  {teams}"
            f"  draft={self.draft_status or '?'}"
        )


class YahooLeaguePuller:
    """Thin, well-behaved wrapper over ``yfpy`` for the pulls this project needs."""

    def __init__(
        self,
        season: int,
        league_id: str | int = "0",
        *,
        browser_callback: Optional[bool] = None,
        env_path: Optional[Path] = None,
        request_pause: float = 0.4,
    ):
        try:
            from yfpy.query import YahooFantasySportsQuery
        except ImportError as exc:  # pragma: no cover
            raise YahooAuthError(
                "yfpy is not installed. `pip install yfpy` (or use .venv)."
            ) from exc

        self.season = int(season)
        self.league_id = str(league_id)
        self.request_pause = request_pause

        env_path = Path(env_path) if env_path else env_file()
        ensure(env_path.parent)
        self._load_env(env_path)

        if not os.environ.get("YAHOO_CONSUMER_KEY") or not os.environ.get(
            "YAHOO_CONSUMER_SECRET"
        ):
            raise YahooAuthError(
                "Missing YAHOO_CONSUMER_KEY / YAHOO_CONSUMER_SECRET.\n"
                f"Create an app at https://developer.yahoo.com/apps/ (Fantasy\n"
                f"Sports, Read permission, redirect URI 'oob'), then write them to\n"
                f"  {env_path}\n"
                "as:\n"
                "  YAHOO_CONSUMER_KEY=...\n"
                "  YAHOO_CONSUMER_SECRET=...\n"
            )

        # On Colab (or any headless runtime) the localhost redirect cannot work,
        # so fall back to out-of-band code entry.
        if browser_callback is None:
            browser_callback = not is_colab() and sys.stdin.isatty()

        self.query = YahooFantasySportsQuery(
            league_id=self.league_id,
            game_code=GAME_CODE,
            yahoo_consumer_key=os.environ["YAHOO_CONSUMER_KEY"],
            yahoo_consumer_secret=os.environ["YAHOO_CONSUMER_SECRET"],
            env_file_location=env_path.parent,
            save_token_data_to_env_file=True,
            browser_callback=bool(browser_callback),
        )

    @staticmethod
    def _load_env(env_path: Path) -> None:
        """Load KEY=VALUE pairs from ``env_path`` into os.environ (no overwrite)."""
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def _pause(self) -> None:
        if self.request_pause:
            time.sleep(self.request_pause)

    # -- discovery ---------------------------------------------------------

    def discover_leagues(self) -> List[LeagueRef]:
        """List the leagues this Yahoo account belongs to, for ``season``.

        Saves you from digging the league id out of a Yahoo URL, and surfaces the
        fact that Yahoo league ids are per-season.
        """
        game_key = self.query.get_game_key_by_season(self.season)
        self._pause()
        leagues = self.query.get_user_leagues_by_game_key(game_key)

        refs: List[LeagueRef] = []
        for league in leagues or []:
            refs.append(
                LeagueRef(
                    league_id=str(_get(league, "league_id", default="?")),
                    name=str(_get(league, "name", default="?")),
                    season=int(_get(league, "season", default=self.season) or self.season),
                    num_teams=_get(league, "num_teams"),
                    league_key=_get(league, "league_key"),
                    draft_status=_get(league, "draft_status"),
                )
            )
        return refs

    # -- pulls -------------------------------------------------------------

    def pull_settings(self) -> Dict[str, Any]:
        """League settings plus a stat_id -> name map, so scoring is readable."""
        settings = _as_dict(self.query.get_league_settings())
        self._pause()
        metadata = _as_dict(self.query.get_league_metadata())
        self._pause()

        # Scoring modifiers arrive as {stat_id: value}; without the category map
        # the config is a wall of opaque integers.
        stat_names: Dict[str, Dict[str, Any]] = {}
        try:
            game_id = int(self.query.get_game_key_by_season(self.season).split(".")[0])
            self._pause()
            categories = _as_dict(
                self.query.get_game_stat_categories_by_game_id(game_id)
            )
            for stat in _iter_stats(categories):
                sid = str(_get(stat, "stat_id", default=""))
                if sid:
                    stat_names[sid] = {
                        "name": _get(stat, "name"),
                        "display_name": _get(stat, "display_name"),
                        "position_types": _get(stat, "position_types"),
                    }
        except Exception as exc:  # pragma: no cover - non-fatal enrichment
            stat_names = {"_error": str(exc)}

        return {
            "season": self.season,
            "league_id": self.league_id,
            "metadata": metadata,
            "settings": settings,
            "stat_categories": stat_names,
        }

    def pull_draft(self) -> pd.DataFrame:
        """Every draft pick. This is the opponent-model calibration data."""
        results = self.query.get_league_draft_results()
        self._pause()
        rows = []
        for pick in results or []:
            rows.append(
                {
                    "pick": _get(pick, "pick"),
                    "round": _get(pick, "round"),
                    "team_key": _get(pick, "team_key"),
                    "player_key": _get(pick, "player_key"),
                    "cost": _get(pick, "cost"),
                }
            )
        return pd.DataFrame(rows)

    def pull_teams(self) -> pd.DataFrame:
        teams = self.query.get_league_teams()
        self._pause()
        rows = []
        for team in teams or []:
            managers = _as_dict(_get(team, "managers", default=[])) or []
            if isinstance(managers, dict):
                managers = [managers]
            nicknames = [
                str(_get(m.get("manager", m) if isinstance(m, dict) else m, "nickname",
                         default="?"))
                for m in managers
            ]
            rows.append(
                {
                    "team_id": _get(team, "team_id"),
                    "team_key": _get(team, "team_key"),
                    "name": _get(team, "name"),
                    "managers": "|".join(nicknames),
                    "is_owned_by_current_login": bool(
                        _get(team, "is_owned_by_current_login", default=0)
                    ),
                    "draft_position": _get(team, "draft_position"),
                    "number_of_moves": _get(team, "number_of_moves"),
                    "number_of_trades": _get(team, "number_of_trades"),
                }
            )
        return pd.DataFrame(rows)

    def pull_standings(self) -> pd.DataFrame:
        standings = self.query.get_league_standings()
        self._pause()
        teams = _get(standings, "teams", default=[]) or []
        rows = []
        for team in teams:
            ts = _get(team, "team_standings")
            outcome = _get(ts, "outcome_totals") if ts is not None else None
            rows.append(
                {
                    "team_id": _get(team, "team_id"),
                    "name": _get(team, "name"),
                    "rank": _get(ts, "rank") if ts is not None else None,
                    "wins": _get(outcome, "wins") if outcome is not None else None,
                    "losses": _get(outcome, "losses") if outcome is not None else None,
                    "ties": _get(outcome, "ties") if outcome is not None else None,
                    "points_for": _get(ts, "points_for") if ts is not None else None,
                    "points_against": (
                        _get(ts, "points_against") if ts is not None else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def pull_matchups(self, weeks: Iterable[int]) -> pd.DataFrame:
        """Per-week H2H results. One row per team per week."""
        rows = []
        for week in weeks:
            try:
                matchups = self.query.get_league_matchups_by_week(week)
            except Exception as exc:
                print(f"  week {week}: {type(exc).__name__}: {exc}")
                continue
            self._pause()
            for matchup in matchups or []:
                teams = _get(matchup, "teams", default=[]) or []
                points = [
                    _to_float(_get(_get(t, "team_points"), "total"))
                    for t in teams
                ]
                for i, team in enumerate(teams):
                    opponent = teams[1 - i] if len(teams) == 2 else None
                    rows.append(
                        {
                            "week": week,
                            "team_id": _get(team, "team_id"),
                            "team_name": _get(team, "name"),
                            "points": points[i] if i < len(points) else None,
                            "opponent_id": (
                                _get(opponent, "team_id") if opponent is not None else None
                            ),
                            "opponent_points": (
                                points[1 - i] if len(points) == 2 else None
                            ),
                            "projected": _to_float(
                                _get(_get(team, "team_projected_points"), "total")
                            ),
                            "is_playoffs": _get(matchup, "is_playoffs"),
                            "is_consolation": _get(matchup, "is_consolation"),
                            "winner_team_key": _get(matchup, "winner_team_key"),
                        }
                    )
        return pd.DataFrame(rows)

    def pull_weekly_rosters(
        self, team_ids: Iterable[int], weeks: Iterable[int], out_path: Path
    ) -> pd.DataFrame:
        """Who was STARTED each week, per team. Slow: one request per team-week.

        Writes incrementally and resumes from ``out_path``, because 12 teams x 17
        weeks is ~200 requests and Colab will disconnect partway through at least
        once.
        """
        team_ids = list(team_ids)
        weeks = list(weeks)

        done: set[tuple[int, int]] = set()
        existing = pd.DataFrame()
        if out_path.exists():
            existing = pd.read_csv(out_path)
            if {"week", "team_id"}.issubset(existing.columns):
                done = set(
                    zip(existing["week"].astype(int), existing["team_id"].astype(int))
                )
                print(f"  resuming: {len(done)} team-weeks already pulled")

        rows: List[Dict[str, Any]] = []
        total = len(team_ids) * len(weeks)
        for n, (week, team_id) in enumerate(
            ((w, t) for w in weeks for t in team_ids), start=1
        ):
            if (int(week), int(team_id)) in done:
                continue
            try:
                roster = self.query.get_team_roster_by_week(team_id, chosen_week=week)
            except Exception as exc:
                print(f"  team {team_id} week {week}: {type(exc).__name__}: {exc}")
                continue
            self._pause()

            for player in _iter_players(_as_dict(roster)):
                selected = _get(player, "selected_position")
                if isinstance(selected, dict):
                    selected = selected.get("position", selected.get("selected_position"))
                rows.append(
                    {
                        "week": week,
                        "team_id": team_id,
                        "player_key": _get(player, "player_key"),
                        "player_name": _player_name(player),
                        "position": _get(player, "primary_position", "display_position"),
                        "selected_position": selected,
                        "is_starter": selected not in (None, "BN", "IR"),
                    }
                )

            if n % 12 == 0 or n == total:
                print(f"  rosters {n}/{total}")
                _append_csv(out_path, rows, existing)
                existing = pd.read_csv(out_path)
                rows = []

        if rows:
            _append_csv(out_path, rows, existing)

        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _player_name(player: Any) -> Optional[str]:
    name = _get(player, "name")
    if isinstance(name, dict):
        return name.get("full") or name.get("first")
    return name if isinstance(name, str) else _get(player, "player_name")


def _iter_stats(node: Any) -> Iterable[Any]:
    """Yield stat dicts from the nested stat-categories payload."""
    if isinstance(node, dict):
        if "stat_id" in node:
            yield node
        for value in node.values():
            yield from _iter_stats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_stats(value)


def _iter_players(node: Any) -> Iterable[Dict[str, Any]]:
    """Yield player dicts from a nested roster payload."""
    if isinstance(node, dict):
        if "player_key" in node:
            yield node
            return
        for value in node.values():
            yield from _iter_players(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_players(value)


def _append_csv(path: Path, rows: List[Dict[str, Any]], existing: pd.DataFrame) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if not existing.empty:
        frame = pd.concat([existing, frame], ignore_index=True)
    ensure(path.parent)
    frame.to_csv(path, index=False)


def normalize_config(settings_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce the raw Yahoo payload to the fields the rest of the code needs.

    Deliberately shallow -- it keeps the raw payload alongside so nothing is lost
    if a field turns out to matter later.
    """
    settings = settings_payload.get("settings") or {}
    metadata = settings_payload.get("metadata") or {}
    stat_names = settings_payload.get("stat_categories") or {}

    roster_slots: Dict[str, int] = {}
    for slot in _as_list(settings.get("roster_positions")):
        position = _get(slot, "position")
        count = _get(slot, "count", default=0)
        if position:
            roster_slots[str(position)] = roster_slots.get(str(position), 0) + int(
                count or 0
            )

    scoring: Dict[str, Any] = {}
    for modifier in _as_list(settings.get("stat_modifiers")):
        for stat in _iter_stats(modifier):
            sid = str(_get(stat, "stat_id", default=""))
            value = _get(stat, "value")
            if not sid or value is None:
                continue
            info = stat_names.get(sid) or {}
            scoring[sid] = {
                "value": _to_float(value),
                "name": info.get("name"),
                "display_name": info.get("display_name"),
            }

    return {
        "source": "yahoo",
        "season": settings_payload.get("season"),
        "league_id": settings_payload.get("league_id"),
        "name": _get(metadata, "name"),
        "num_teams": _to_int(_get(metadata, "num_teams")),
        "roster_slots": roster_slots,
        "scoring": scoring,
        "playoff_start_week": _to_int(_get(settings, "playoff_start_week")),
        "num_playoff_teams": _to_int(_get(settings, "num_playoff_teams")),
        "start_week": _to_int(_get(metadata, "start_week")),
        "end_week": _to_int(_get(metadata, "end_week")),
        "scoring_type": _get(metadata, "scoring_type"),
        "is_auction": bool(_get(settings, "uses_faab", default=False)),
        "raw": settings_payload,
    }


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Credential setup / diagnosis
# ---------------------------------------------------------------------------

CREDENTIAL_KEYS = ("YAHOO_CONSUMER_KEY", "YAHOO_CONSUMER_SECRET")
TOKEN_KEYS = (
    "YAHOO_ACCESS_TOKEN",
    "YAHOO_REFRESH_TOKEN",
    "YAHOO_TOKEN_TIME",
    "YAHOO_TOKEN_TYPE",
)


def read_env(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse the cache .env into a dict (does not touch os.environ)."""
    path = path or env_file()
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env(updates: Dict[str, str], path: Optional[Path] = None) -> Path:
    """Merge ``updates`` into the cache .env, preserving other keys.

    Written 0600 -- this file holds an OAuth refresh token with read access to
    the account's fantasy data.
    """
    path = path or env_file()
    ensure(path.parent)
    current = read_env(path)
    current.update({k: v for k, v in updates.items() if v})

    body = "\n".join(f"{k}={v}" for k, v in sorted(current.items()))
    path.write_text(
        "# Yahoo Fantasy API credentials + cached OAuth token.\n"
        "# Managed by `python -m src.data.yahoo_league`. Never commit this.\n"
        f"{body}\n"
    )
    path.chmod(0o600)
    return path


def doctor(path: Optional[Path] = None) -> int:
    """Diagnose the OAuth setup and print the single next action.

    OAuth has several failure modes that all surface as the same unhelpful
    error, so this separates them.
    """
    path = path or env_file()
    values = read_env(path)

    print(f"env file : {path}{'' if path.exists() else '   (does not exist yet)'}")
    if path.exists():
        print(f"perms    : {oct(path.stat().st_mode & 0o777)}")

    have_creds = all(values.get(k) for k in CREDENTIAL_KEYS)
    have_token = bool(values.get("YAHOO_REFRESH_TOKEN"))

    for key in CREDENTIAL_KEYS:
        value = values.get(key)
        shown = f"{value[:6]}…({len(value)} chars)" if value else "MISSING"
        print(f"{key:22s}: {shown}")
    for key in TOKEN_KEYS:
        print(f"{key:22s}: {'present' if values.get(key) else '-'}")

    print()
    if not have_creds:
        print("NEXT: you need a Yahoo app (2 minutes, free).")
        print("  1. https://developer.yahoo.com/apps/create/")
        print("     Application Type : Installed Application")
        print("     Redirect URI     : oob          <- must be exactly this")
        print("     API Permissions  : Fantasy Sports -> Read")
        print("  2. Copy the Client ID and Client Secret it shows you, then:")
        print("     python -m src.data.yahoo_league --set-credentials <ID> <SECRET>")
        print("  3. python -m src.data.yahoo_league --auth")
        return 1

    if not have_token:
        print("NEXT: credentials are set, but you have not authorized yet.")
        print("  python -m src.data.yahoo_league --auth")
        print("  (headless/Colab: add --no-browser to paste the code by hand)")
        return 1

    print("NEXT: looks complete. Verify with:")
    print("  python -m src.data.yahoo_league --discover --season 2025")
    print()
    print("If that fails with a 401/invalid_grant, the refresh token was revoked")
    print("(happens when you change the app's permissions). Clear it and redo:")
    print("  python -m src.data.yahoo_league --reset-token")
    print("  python -m src.data.yahoo_league --auth")
    return 0


def reset_token(path: Optional[Path] = None) -> Path:
    """Drop cached token fields, keeping the consumer key/secret.

    The fix for ``invalid_grant`` -- a stale refresh token is not recoverable,
    it has to be re-issued by consenting again.
    """
    path = path or env_file()
    values = read_env(path)
    removed = [k for k in (*TOKEN_KEYS, "YAHOO_GUID") if values.pop(k, None) is not None]

    ensure(path.parent)
    body = "\n".join(f"{k}={v}" for k, v in sorted(values.items()))
    path.write_text(
        "# Yahoo Fantasy API credentials + cached OAuth token.\n"
        "# Managed by `python -m src.data.yahoo_league`. Never commit this.\n"
        f"{body}\n"
    )
    path.chmod(0o600)
    print(f"cleared {len(removed)} token field(s) from {path}")
    print("now run: python -m src.data.yahoo_league --auth")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def pull_all(
    season: int,
    league_id: str,
    *,
    with_rosters: bool = False,
    browser_callback: Optional[bool] = None,
) -> Path:
    """Pull one season of one league to ``data/league/yahoo_<id>_<season>/``."""
    puller = YahooLeaguePuller(
        season, league_id, browser_callback=browser_callback
    )
    out = ensure(league_dir(season, league_id))
    print(f"→ {out}")

    print("settings...")
    settings_payload = puller.pull_settings()
    (out / "settings.json").write_text(json.dumps(settings_payload, indent=2, default=str))

    config = normalize_config(settings_payload)
    (out / "league_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(
        f"  {config['num_teams']} teams | roster {config['roster_slots']} | "
        f"scoring_type={config['scoring_type']} | "
        f"playoffs W{config['playoff_start_week']} x{config['num_playoff_teams']}"
    )

    print("teams...")
    teams = puller.pull_teams()
    teams.to_csv(out / "managers.csv", index=False)
    print(f"  {len(teams)} teams")

    print("draft...")
    draft = puller.pull_draft()
    draft.to_csv(out / "draft_picks.csv", index=False)
    print(f"  {len(draft)} picks")

    print("standings...")
    standings = puller.pull_standings()
    standings.to_csv(out / "standings.csv", index=False)
    print(f"  {len(standings)} rows")

    start = config.get("start_week") or 1
    end = config.get("end_week") or 17
    print(f"matchups (weeks {start}-{end})...")
    matchups = puller.pull_matchups(range(int(start), int(end) + 1))
    matchups.to_csv(out / "weekly_matchups.csv", index=False)
    print(f"  {len(matchups)} team-weeks")

    if with_rosters and not teams.empty:
        print("weekly rosters (slow, resumable)...")
        puller.pull_weekly_rosters(
            teams["team_id"].dropna().astype(int).tolist(),
            range(int(start), int(end) + 1),
            out / "weekly_rosters.csv",
        )

    print(f"\ndone: {out}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull Yahoo fantasy league config and history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--league-id", type=str, default=None)
    parser.add_argument(
        "--discover", action="store_true", help="list your leagues for --season"
    )
    parser.add_argument(
        "--auth", action="store_true", help="run the OAuth consent flow and exit"
    )
    parser.add_argument(
        "--rosters",
        action="store_true",
        help="also pull per-week started lineups (~200 requests, resumable)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="out-of-band code entry instead of a localhost redirect (Colab)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="diagnose OAuth setup and print the next action",
    )
    parser.add_argument(
        "--set-credentials",
        nargs=2,
        metavar=("CLIENT_ID", "CLIENT_SECRET"),
        help="save Yahoo app credentials to the cache .env (chmod 600)",
    )
    parser.add_argument(
        "--reset-token",
        action="store_true",
        help="clear the cached OAuth token, keeping credentials (fixes invalid_grant)",
    )
    args = parser.parse_args(argv)

    if args.set_credentials:
        client_id, client_secret = args.set_credentials
        path = write_env(
            {
                "YAHOO_CONSUMER_KEY": client_id,
                "YAHOO_CONSUMER_SECRET": client_secret,
            }
        )
        print(f"saved credentials to {path}")
        print("next: python -m src.data.yahoo_league --auth")
        return 0

    if args.reset_token:
        reset_token()
        return 0

    if args.doctor:
        return doctor()

    browser = False if args.no_browser else None

    try:
        if args.auth or args.discover or not args.league_id:
            puller = YahooLeaguePuller(
                args.season, args.league_id or "0", browser_callback=browser
            )
            if args.auth:
                user = puller.query.get_current_user()
                print(f"authenticated as: {_get(user, 'nickname', 'guid', default=user)}")
                print(f"tokens cached in: {env_file()}")
                return 0

            leagues = puller.discover_leagues()
            if not leagues:
                print(f"no leagues found for {args.season}.")
                return 1
            print(f"leagues for {args.season}:")
            for ref in leagues:
                print(f"  {ref}")
            if not args.league_id:
                print("\nre-run with --league-id <id> to pull one.")
                return 0

        pull_all(
            args.season,
            args.league_id,
            with_rosters=args.rosters,
            browser_callback=browser,
        )
        return 0

    except YahooAuthError as exc:
        print(f"\nauth error:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
