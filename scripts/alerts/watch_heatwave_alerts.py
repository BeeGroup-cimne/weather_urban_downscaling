#!/usr/bin/env python3
"""Continuously regenerate heatwave alerts when new prediction maps arrive."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.alerts.generate_heatwave_alerts import LEVEL_RANK
from scripts.alerts.generate_heatwave_alerts import main as generate_alerts


STOP_REQUESTED = False


@dataclass(frozen=True)
class FileState:
    path: str
    size: int
    mtime_ns: int


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _stable_files(prediction_dir: Path, pattern: str, stable_seconds: float) -> list[Path]:
    now = time.time()
    files = []
    for path in sorted(prediction_dir.glob(pattern)):
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size <= 0:
            continue
        if now - stat.st_mtime < stable_seconds:
            continue
        files.append(path)
    return files


def _signature(files: list[Path]) -> tuple[FileState, ...]:
    states = []
    for path in files:
        stat = path.stat()
        states.append(FileState(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(states)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _notify_webhook(args: argparse.Namespace, out_dir: Path, last_key: str | None) -> str | None:
    webhook_url = args.webhook_url or os.getenv("ALERT_WEBHOOK_URL", "")
    if not webhook_url:
        return last_key

    latest_path = out_dir / "latest_alert.json"
    if not latest_path.exists():
        return last_key

    with latest_path.open("r", encoding="utf-8") as f:
        latest = json.load(f)

    level = str(latest.get("alert_level", "normal"))
    if LEVEL_RANK.get(level, 0) < LEVEL_RANK.get(args.notify_min_level, LEVEL_RANK["warning"]):
        return last_key

    key = f"{latest.get('time')}:{level}:{latest.get('is_active_heatwave_event')}"
    if key == last_key:
        return last_key

    body = json.dumps(
        {
            "type": "heatwave_alert",
            "alert": latest,
            "source": "weather_urban_downscaling",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(args.webhook_timeout_seconds)) as response:
        if response.status >= 300:
            raise RuntimeError(f"webhook returned HTTP {response.status}")
    print(f"webhook notification sent for {key}")
    return key


def _build_generate_args(args: argparse.Namespace) -> list[str]:
    generate_args = [
        "--prediction-dir",
        args.prediction_dir,
        "--pattern",
        args.pattern,
        "--time-regex",
        args.time_regex,
        "--out-dir",
        args.out_dir,
        "--watch-fraction",
        str(args.watch_fraction),
        "--warning-fraction",
        str(args.warning_fraction),
        "--severe-fraction",
        str(args.severe_fraction),
        "--min-duration-days",
        str(args.min_duration_days),
    ]

    if args.threshold_celsius is not None:
        generate_args.extend(["--threshold-celsius", str(args.threshold_celsius)])
    elif args.threshold_map:
        generate_args.extend(["--threshold-map", args.threshold_map])
        if args.threshold_var:
            generate_args.extend(["--threshold-var", args.threshold_var])
    else:
        generate_args.append("--derive-threshold-from-cache")
        generate_args.extend(["--base-start", args.base_start])
        generate_args.extend(["--base-end", args.base_end])
        generate_args.extend(["--months", args.months])
        generate_args.extend(["--pctl", str(args.pctl)])

    return generate_args


def _run_generator(generate_args: list[str]) -> None:
    try:
        rc = generate_alerts(generate_args)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    if rc != 0:
        raise RuntimeError(f"alert generation exited with code {rc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="experiments/predictions")
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument(
        "--time-regex",
        default=r"(\d{4}[-_]\d{2}[-_]\d{2}[T_]\d{2}[_:]\d{2}(?:[_:]\d{2})?)",
    )
    parser.add_argument("--threshold-celsius", type=float, default=None)
    parser.add_argument("--threshold-map", default="")
    parser.add_argument("--threshold-var", default="")
    parser.add_argument("--base-start", default="2017-01-01")
    parser.add_argument("--base-end", default="2018-01-01")
    parser.add_argument("--months", default="6,7,8,9")
    parser.add_argument("--pctl", type=float, default=0.95)
    parser.add_argument("--watch-fraction", type=float, default=0.05)
    parser.add_argument("--warning-fraction", type=float, default=0.10)
    parser.add_argument("--severe-fraction", type=float, default=0.25)
    parser.add_argument("--min-duration-days", type=int, default=3)
    parser.add_argument("--out-dir", default="experiments/alerts/latest")
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--stable-seconds", type=float, default=10.0)
    parser.add_argument("--run-on-start", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=0, help="Stop after N loops; 0 means run forever.")
    parser.add_argument("--webhook-url", default=os.getenv("ALERT_WEBHOOK_URL", ""))
    parser.add_argument("--notify-min-level", default=os.getenv("ALERT_NOTIFY_MIN_LEVEL", "warning"), choices=list(LEVEL_RANK))
    parser.add_argument("--webhook-timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    prediction_dir = Path(args.prediction_dir)
    out_dir = Path(args.out_dir)
    status_path = out_dir / "watcher_status.json"
    last_signature: tuple[FileState, ...] | None = None
    last_run_time = None
    last_error = None
    last_notification_key = None
    iterations = 0

    print(f"watching predictions: {prediction_dir} pattern={args.pattern}")
    while not STOP_REQUESTED:
        iterations += 1
        try:
            files = _stable_files(prediction_dir, args.pattern, args.stable_seconds)
            signature = _signature(files)
            should_run = bool(files) and (signature != last_signature or (args.run_on_start and last_signature is None))

            if should_run:
                print(f"prediction change detected: {len(files)} file(s)")
                _run_generator(_build_generate_args(args))
                last_notification_key = _notify_webhook(args, out_dir, last_notification_key)
                last_signature = signature
                last_run_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                last_error = None

            _write_status(
                status_path,
                {
                    "status": "stopping" if STOP_REQUESTED else "watching",
                    "prediction_dir": str(prediction_dir),
                    "pattern": args.pattern,
                    "stable_file_count": len(files),
                    "last_run_time": last_run_time,
                    "last_error": last_error,
                    "last_notification_key": last_notification_key,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as exc:  # pragma: no cover - long-running operational path.
            last_error = str(exc)
            _write_status(
                status_path,
                {
                    "status": "error",
                    "prediction_dir": str(prediction_dir),
                    "pattern": args.pattern,
                    "last_run_time": last_run_time,
                    "last_error": last_error,
                    "last_notification_key": last_notification_key,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            print(f"alert watcher error: {exc}", file=sys.stderr)

        if STOP_REQUESTED:
            break
        if args.max_iterations and iterations >= int(args.max_iterations):
            break
        time.sleep(max(1.0, float(args.interval_seconds)))

    _write_status(
        status_path,
        {
            "status": "stopped",
            "prediction_dir": str(prediction_dir),
            "pattern": args.pattern,
            "last_run_time": last_run_time,
            "last_error": last_error,
            "last_notification_key": last_notification_key,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print("alert watcher stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
