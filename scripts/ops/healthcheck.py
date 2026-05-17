#!/usr/bin/env python3
"""Lightweight container/runtime health check for batch jobs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _check_path(path: Path, *, writable: bool = False) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"missing path: {path}")
        return errors
    if writable and not os.access(path, os.W_OK):
        errors.append(f"path is not writable: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Check the container runtime contract without loading training data.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Also require the known production data inputs to exist.",
    )
    args = parser.parse_args()

    root = Path(os.getenv("APP_ROOT", "/app"))
    errors: list[str] = []

    errors.extend(_check_path(root / "src"))
    errors.extend(_check_path(root / "config"))
    errors.extend(_check_path(root / "scripts"))
    errors.extend(_check_path(root / "data"))
    errors.extend(_check_path(root / "experiments", writable=True))

    if args.runtime:
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import xarray  # noqa: F401
        except Exception as exc:  # pragma: no cover - exercised in container.
            errors.append(f"runtime import failed: {exc}")

    if args.strict_data:
        required = [
            root / "data/processed/estaciones_interpoladas_final.nc",
            root / "data/processed/era5land/lr_2017.grib",
            root / "data/processed/weather_static_FINAL_stations.zarr",
        ]
        for path in required:
            errors.extend(_check_path(path))

    if errors:
        print("healthcheck failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("healthcheck ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
