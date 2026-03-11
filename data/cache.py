"""
Simple disk-based JSON cache to avoid hammering rate-limited APIs.
Each cache entry is a JSON file in .cache/ with a stored timestamp.
"""

import json
import os
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> Path:
    # sanitize key to a safe filename
    safe = key.replace("/", "__").replace(":", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe}.json"


def get_cached(key: str, ttl_seconds: float):
    """
    Return cached data if it exists and is fresher than ttl_seconds.
    Returns None if missing or stale.
    """
    path = _cache_path(key)
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            entry = json.load(f)
        cached_at = entry.get("_cached_at", 0)
        if time.time() - cached_at <= ttl_seconds:
            return entry.get("data")
    except (json.JSONDecodeError, OSError):
        return None

    return None


def get_stale(key: str):
    """
    Return cached data regardless of age, or None if no cache exists.
    Useful as a fallback when an API call fails.
    """
    path = _cache_path(key)
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            entry = json.load(f)
        return entry.get("data")
    except (json.JSONDecodeError, OSError):
        return None


def set_cached(key: str, data):
    """Save data to disk cache with current timestamp."""
    _ensure_cache_dir()
    path = _cache_path(key)
    entry = {
        "_cached_at": time.time(),
        "data": data,
    }
    try:
        with open(path, "w") as f:
            json.dump(entry, f)
    except (OSError, TypeError):
        pass  # non-critical — just skip caching
