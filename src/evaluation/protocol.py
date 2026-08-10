"""The holdout is only worth something if the code refuses to spend it.

2024 and 2025 are the final test set. The pre-registered budget is at most
``MAX_GATE_TOUCHES`` evaluations against them, for the whole project -- after
that the seasons are burned, because a number you can look at repeatedly is a
number you can tune against, and it stops being held out.

Until now that separation lived entirely in human discipline: nothing stopped a
casual ``--season 2025`` from quietly consuming a touch. This module makes it
mechanical. A run declares its protocol up front:

* ``tune``  -- 2022/2023, unlimited, and *forbidden* on a gate season.
* ``gate``  -- 2024/2025 only, and only under a name already present in
  ``gate_registry.json`` with the hypothesis written down.

The registry is checked in, so pre-registration is a commit that predates the
result. Writing the hypothesis *after* seeing the number is the failure mode
this is built to prevent, and a diff makes that visible.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..data.paths import REPO_ROOT

TUNING_SEASONS = (2022, 2023)
GATE_SEASONS = (2024, 2025)

# Total across the whole registry, not per experiment: three looks at the
# holdout is the budget the project pre-registered, and splitting it across
# names would be the obvious way to launder extra looks.
MAX_GATE_TOUCHES = 3

PROTOCOLS = ("tune", "gate")

REGISTRY_PATH = REPO_ROOT / "gate_registry.json"

EMPTY_REGISTRY: Dict[str, Dict] = {"experiments": {}}


class GateError(Exception):
    """A run would violate the holdout protocol. Raised instead of proceeding."""


def _path(registry_path: Optional[Path] = None) -> Path:
    return Path(registry_path) if registry_path is not None else REGISTRY_PATH


def load_registry(path: Optional[Path] = None) -> Dict:
    """Read the gate registry, creating an empty one if it does not exist.

    Creating the file is safe; auto-registering an experiment would not be, so
    a missing file yields no experiments rather than a permissive default.
    """
    target = _path(path)
    if not target.exists():
        _write_registry(EMPTY_REGISTRY, target)
        return json.loads(json.dumps(EMPTY_REGISTRY))

    with open(target, "r") as handle:
        registry = json.load(handle)
    registry.setdefault("experiments", {})
    return registry


def _write_registry(registry: Dict, path: Path) -> None:
    """Write-then-rename, so an interrupted write cannot truncate the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def touches_spent(registry: Dict) -> int:
    """Touches consumed across every registered experiment."""
    return sum(
        len(entry.get("touches", []))
        for entry in registry.get("experiments", {}).values()
    )


def check(
    season: int,
    *,
    protocol: str,
    register: Optional[str] = None,
    registry_path: Optional[Path] = None,
) -> None:
    """Raise ``GateError`` unless this run is allowed to touch ``season``.

    Called before any work happens, so a violation costs nothing but a message.
    """
    if protocol not in PROTOCOLS:
        raise GateError(
            f"unknown protocol {protocol!r}; expected one of {list(PROTOCOLS)}"
        )

    if protocol == "tune":
        if season in GATE_SEASONS:
            raise GateError(
                f"{season} is a held-out gate season and cannot be run under "
                f"protocol='tune'. Tune on {list(TUNING_SEASONS)}. If you really "
                f"mean to spend a holdout touch, pre-register the experiment in "
                f"{REGISTRY_PATH.name} and pass protocol='gate' with register=<name>."
            )
        return

    if season not in GATE_SEASONS:
        raise GateError(
            f"protocol='gate' is only for the held-out seasons {list(GATE_SEASONS)}; "
            f"{season} is not one. A gate run on tuning data measures nothing."
        )

    if not register:
        raise GateError(
            "protocol='gate' requires register=<name> naming a pre-registered "
            f"experiment in {REGISTRY_PATH.name}."
        )

    registry = load_registry(registry_path)
    experiments = registry.get("experiments", {})

    if register not in experiments:
        known = sorted(experiments) or ["<none registered>"]
        shape = json.dumps(
            {
                "experiments": {
                    register: {
                        "hypothesis": "what you expect to see, and why",
                        "registered": datetime.now().strftime("%Y-%m-%d"),
                        "touches": [],
                    }
                }
            }
        )
        raise GateError(
            f"{register!r} is not in {REGISTRY_PATH.name} (have: {known}). Add an "
            f"entry describing the hypothesis you are testing and commit it "
            f"BEFORE running -- a hypothesis written after the result is not a "
            f"hypothesis. Shape: {shape}"
        )

    spent = touches_spent(registry)
    if spent >= MAX_GATE_TOUCHES:
        used = {
            name: len(entry.get("touches", []))
            for name, entry in experiments.items()
            if entry.get("touches")
        }
        raise GateError(
            f"the holdout budget is spent: {spent}/{MAX_GATE_TOUCHES} touches "
            f"already recorded ({used}). 2024/2025 are burned; any further "
            f"evaluation against them is tuning, not a test."
        )


def record_touch(
    name: str,
    season: int,
    registry_path: Optional[Path] = None,
    *,
    note: Optional[str] = None,
) -> None:
    """Append one spent touch to ``name`` and rewrite the registry atomically.

    Recorded only after a gate run produces results, so a crashed run does not
    cost a touch.
    """
    path = _path(registry_path)
    registry = load_registry(path)
    experiments = registry.setdefault("experiments", {})

    if name not in experiments:
        raise GateError(
            f"cannot record a touch for unregistered experiment {name!r}"
        )

    touch = {"season": season, "at": datetime.now().isoformat(timespec="seconds")}
    if note:
        touch["note"] = note
    experiments[name].setdefault("touches", []).append(touch)

    _write_registry(registry, path)
