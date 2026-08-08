"""Filesystem layout, resolved the same way locally and on Colab.

Colab runtimes are ephemeral: anything not written to Drive is gone on restart,
and re-downloading seven seasons of nflverse parquet every session is slow enough
that people start hardcoding paths in notebook cells. That is how this repo ended
up with three divergent copies of the same trainer.

So there is exactly one knob -- FF_CACHE_DIR -- and it has a sensible default in
both environments.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_COLAB_CACHE = Path("/content/drive/MyDrive/fantasy/cache")


def is_colab() -> bool:
    """True when running inside a Google Colab runtime.

    find_spec on a dotted name imports the parent package, so it raises
    ModuleNotFoundError rather than returning None when `google` is absent --
    which is the normal case off Colab.
    """
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def drive_is_mounted() -> bool:
    """True when Google Drive appears to be mounted at the usual location."""
    return Path("/content/drive/MyDrive").is_dir()


def cache_dir() -> Path:
    """Root for downloaded/derived data that is expensive to rebuild.

    Resolution order:
      1. $FF_CACHE_DIR, if set
      2. Google Drive, on Colab with Drive mounted
      3. <repo>/data/cache

    On Colab without Drive mounted this falls back to local disk and warns --
    it will work, but the cache dies with the runtime.
    """
    override = os.environ.get("FF_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    if is_colab():
        if drive_is_mounted():
            return DEFAULT_COLAB_CACHE
        import warnings
        warnings.warn(
            "Running on Colab without Google Drive mounted -- the cache will be "
            "lost when this runtime recycles. Mount it with:\n"
            "    from google.colab import drive; drive.mount('/content/drive')\n"
            "or set FF_CACHE_DIR to silence this.",
            stacklevel=2,
        )

    return REPO_ROOT / "data" / "cache"


def ensure(path: Path) -> Path:
    """Create a directory (and parents) and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def league_dir(season: int, league_id: str | int) -> Path:
    """Where a single season's pulled league data lives."""
    return REPO_ROOT / "data" / "league" / f"yahoo_{league_id}_{season}"


def nflverse_dir() -> Path:
    """Cache for nflverse parquet releases."""
    return cache_dir() / "nflverse"


def env_file() -> Path:
    """Location of the .env holding Yahoo credentials and OAuth tokens.

    Kept in the cache dir rather than the repo so that copying the cache to
    Drive carries the token with it -- which is the whole trick for using an
    already-authenticated session from Colab.
    """
    override = os.environ.get("FF_ENV_FILE")
    if override:
        return Path(override).expanduser()
    return cache_dir() / ".env"


def describe() -> str:
    """One-line summary of the resolved environment, for notebook cells."""
    where = "colab" if is_colab() else "local"
    drive = " (drive mounted)" if is_colab() and drive_is_mounted() else ""
    return (
        f"env={where}{drive}\n"
        f"repo={REPO_ROOT}\n"
        f"cache={cache_dir()}\n"
        f"env_file={env_file()}"
    )
