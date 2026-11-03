"""Small, dependency-free helpers for cycle freshness validation."""
from datetime import datetime, timezone
from pathlib import Path


def utc_naive(value):
    """Normalize an ECMWF cycle timestamp for consistent comparisons."""
    if not isinstance(value, datetime):
        raise TypeError("Expected datetime for model cycle")
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(microsecond=0)


def read_previous_cycle(marker):
    """Return previously published cycle or None when not recorded."""
    try:
        raw = Path(marker).read_text(encoding="utf-8").strip()
        return utc_naive(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def is_newer_cycle(candidate, previous):
    """Never overwrite published output with the same or an older cycle."""
    return previous is None or utc_naive(candidate) > utc_naive(previous)
