"""Resolve immutable runtime assets in source checkouts and execution-library wheels."""

from pathlib import Path


def bundled_directory(name: str) -> Path:
    """Prefer wheel resources when present; keep the source-checkout layout otherwise."""
    root = Path(__file__).resolve().parent.parent
    packaged = root / "hermes_resources" / name
    return packaged if packaged.is_dir() else root / name
