#!/usr/bin/env python3
"""
Small utilities for paper figure scripts.
Keeps outputs consistent and avoids duplicated boilerplate.
"""

import os
from datetime import datetime


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def default_fig_dir() -> str:
    return os.path.join("experiments", "figures")


def safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception as exc:
        print(f"❌ matplotlib not available: {exc}")
        print("Install it with: pip install matplotlib")
        return False
