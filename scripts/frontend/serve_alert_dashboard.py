#!/usr/bin/env python3
"""Serve the minimal heat alert dashboard and backend artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend" / "alert-dashboard"
ALERT_DIR = ROOT / "experiments" / "alerts" / "latest"
PREDICTION_DIR = ROOT / "experiments" / "predictions"
CACHE_ZARR = ROOT / "data" / "processed" / "weather_cache.zarr"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size <= 1:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _latest_prediction_path() -> Path | None:
    manifest = PREDICTION_DIR / "manifest.csv"
    rows = _read_csv(manifest)
    written = [row for row in rows if row.get("status", "written") in {"written", "skipped"}]
    if written:
        latest = written[-1]
        path = Path(latest["path"])
        return path if path.is_absolute() else ROOT / path
    candidates = sorted(PREDICTION_DIR.glob("pred_*.npy"))
    return candidates[-1] if candidates else None


def _downsample_array(arr: np.ndarray, max_size: int) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")
    h, w = arr.shape
    step = max(1, int(np.ceil(max(h, w) / float(max_size))))
    return arr[::step, ::step].astype(np.float32)


def _array_payload(path: Path, max_size: int) -> dict:
    arr = _downsample_array(np.load(path), max_size=max_size)
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr, dtype=np.float32)
        min_val = max_val = 0.0
    else:
        min_val = float(np.nanmin(arr))
        max_val = float(np.nanmax(arr))
    return {
        "path": str(path),
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "min": min_val,
        "max": max_val,
        "values": arr.reshape(-1).tolist(),
    }


def _geo_payload(max_size: int) -> dict:
    if not CACHE_ZARR.exists():
        return {}
    ds = xr.open_zarr(CACHE_ZARR, consolidated=True)
    da = ds["hr_target"]
    lat = np.asarray(da.coords.get("latitude_2d", da.coords.get("latitude")).values)
    lon = np.asarray(da.coords.get("longitude_2d", da.coords.get("longitude")).values)
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    lat_ds = _downsample_array(lat, max_size=max_size)
    lon_ds = _downsample_array(lon, max_size=max_size)
    return {
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
        "width": int(lat_ds.shape[1]),
        "height": int(lat_ds.shape[0]),
        "latitudes": lat_ds.reshape(-1).tolist(),
        "longitudes": lon_ds.reshape(-1).tolist(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "HeatAlertDashboard/0.1"

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not_found"}, status=404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "latest_alert_exists": (ALERT_DIR / "latest_alert.json").exists(),
                        "prediction_count": len(list(PREDICTION_DIR.glob("pred_*.npy"))),
                    }
                )
                return
            if path == "/api/latest-alert":
                self._send_json(_read_json(ALERT_DIR / "latest_alert.json"))
                return
            if path == "/api/summary":
                self._send_json(_read_json(ALERT_DIR / "alerts_summary.json"))
                return
            if path == "/api/hourly":
                self._send_json(_read_csv(ALERT_DIR / "alerts_hourly.csv"))
                return
            if path == "/api/daily":
                self._send_json(_read_csv(ALERT_DIR / "alerts_daily.csv"))
                return
            if path == "/api/events":
                self._send_json(_read_csv(ALERT_DIR / "heatwave_events.csv"))
                return
            if path == "/api/map/prediction":
                pred_path = _latest_prediction_path()
                if pred_path is None:
                    self._send_json({"error": "prediction_not_found"}, status=404)
                    return
                max_size = int(query.get("max_size", ["160"])[0])
                payload = _array_payload(pred_path, max_size=max_size)
                payload["geo"] = _geo_payload(max_size=max_size)
                self._send_json(payload)
                return
            if path == "/api/map/exceedance":
                max_size = int(query.get("max_size", ["160"])[0])
                payload = _array_payload(ALERT_DIR / "latest_exceedance.npy", max_size=max_size)
                payload["geo"] = _geo_payload(max_size=max_size)
                self._send_json(payload)
                return

            rel = "index.html" if path == "/" else path.lstrip("/")
            safe_path = (FRONTEND_DIR / rel).resolve()
            if FRONTEND_DIR.resolve() not in safe_path.parents and safe_path != FRONTEND_DIR.resolve():
                self._send_json({"error": "forbidden"}, status=403)
                return
            self._send_file(safe_path)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    server = ThreadingHTTPServer((args.host, int(args.port)), DashboardHandler)
    server.quiet = bool(args.quiet)
    print(f"alert dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping alert dashboard")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
