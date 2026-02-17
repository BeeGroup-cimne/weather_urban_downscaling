#!/usr/bin/env python3
"""
Derive AEMET-style heatwave days/events from gridded station-like data.

Dataset expected (Zarr):
  - dims: time, weatherStation
  - var:  airTemperature (float, typically °C)

Method (AEMET-like, adapted to gridded "stations"):
  1) Compute daily Tmax for each "station"/grid cell.
  2) Build per-station threshold as p95 of Tmax over July-August of a base period.
  3) For a target year, flag "heatwave day" if >= X fraction of stations exceed
     their threshold (default X=0.10).
  4) Group consecutive flagged days into events with min duration >= N (default N=3).

Outputs:
  - thresholds.nc         (per-station threshold)
  - daily_YYYY.csv        (daily exceed fraction + flags)
  - events_YYYY.csv       (event summaries)
  - event_days_YYYY.txt   (dates part of events)
  - event_times_YYYY.txt  (hourly timestamps within event days)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class HeatwaveEvent:
    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: int
    peak_exceed_fraction: float
    mean_exceed_fraction: float


def _as_date_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Normalize to midnight, keep tz-naive.
    return pd.to_datetime(index).tz_localize(None).normalize()


def _events_from_flags(
    dates: pd.DatetimeIndex,
    flagged: np.ndarray,
    exceed_fraction: np.ndarray,
    min_duration_days: int,
) -> list[HeatwaveEvent]:
    events: list[HeatwaveEvent] = []
    in_event = False
    start_i = 0
    event_id = 0

    def _finish(end_i: int):
        nonlocal event_id
        duration = end_i - start_i + 1
        if duration < min_duration_days:
            return
        event_id += 1
        frac_slice = exceed_fraction[start_i : end_i + 1]
        events.append(
            HeatwaveEvent(
                event_id=event_id,
                start=dates[start_i],
                end=dates[end_i],
                duration_days=int(duration),
                peak_exceed_fraction=float(np.nanmax(frac_slice)),
                mean_exceed_fraction=float(np.nanmean(frac_slice)),
            )
        )

    for i, is_hot in enumerate(flagged):
        if bool(is_hot) and not in_event:
            in_event = True
            start_i = i
        if in_event and (not bool(is_hot) or i == len(flagged) - 1):
            end_i = i if bool(is_hot) and i == len(flagged) - 1 else i - 1
            _finish(end_i)
            in_event = False

    return events


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zarr",
        default=os.path.join("data", "raw", "weather_stations.zarr"),
        help="Path to weather stations Zarr.",
    )
    p.add_argument("--var", default="airTemperature", help="Temperature variable name.")
    p.add_argument("--base-start", default="2008-01-01", help="Base period start (inclusive).")
    p.add_argument("--base-end", default="2017-01-01", help="Base period end (exclusive).")
    p.add_argument("--target-year", type=int, default=2017, help="Year to detect events in.")
    p.add_argument("--months", default="7,8", help="Comma-separated months used for threshold (default: 7,8).")
    p.add_argument("--pctl", type=float, default=0.95, help="Percentile for threshold (default: 0.95).")
    p.add_argument("--min-fraction", type=float, default=0.10, help="Min fraction of stations exceeding threshold.")
    p.add_argument("--min-duration", type=int, default=3, help="Min consecutive days.")
    p.add_argument(
        "--time-offset-hours",
        type=float,
        default=0.0,
        help="Shift timestamps by this many hours before daily aggregation (optional).",
    )
    p.add_argument("--out-dir", default=os.path.join("experiments", "heatwaves", "aemet"), help="Output directory.")
    p.add_argument(
        "--subsample-stations",
        type=int,
        default=0,
        help="If >0, randomly subsample this many stations for a fast dry-run.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    months = [int(m.strip()) for m in str(args.months).split(",") if m.strip()]
    if not months:
        raise SystemExit("Invalid --months. Example: --months 7,8")
    if not (0.0 < args.pctl < 1.0):
        raise SystemExit("--pctl must be between 0 and 1.")

    ds = xr.open_zarr(args.zarr, consolidated=False)
    if args.var not in ds:
        raise SystemExit(f"Variable not found: {args.var}. Available: {list(ds.data_vars)}")

    da = ds[args.var]
    if "time" not in da.dims or "weatherStation" not in da.dims:
        raise SystemExit(f"Unexpected dims: {da.dims}. Expected ('time','weatherStation').")

    if args.subsample_stations and args.subsample_stations > 0:
        rng = np.random.default_rng(args.seed)
        n = int(args.subsample_stations)
        if n >= da.sizes["weatherStation"]:
            print("ℹ️ subsample-stations >= total; using all stations.")
        else:
            idx = rng.choice(da.sizes["weatherStation"], size=n, replace=False)
            idx = np.sort(idx)
            da = da.isel(weatherStation=idx)

    base_start = pd.Timestamp(args.base_start)
    base_end = pd.Timestamp(args.base_end)
    year_start = pd.Timestamp(f"{args.target_year}-01-01")
    year_end = pd.Timestamp(f"{args.target_year + 1}-01-01")

    if args.time_offset_hours:
        shift = np.timedelta64(int(args.time_offset_hours * 3600), "s")
        da = da.assign_coords(time=da["time"] + shift)

    # Thresholds from base period, months Jul-Aug (default) across multiple years.
    # Compute daily Tmax only for the base period to avoid unnecessary work.
    base_hourly = da.sel(time=(da["time"] >= base_start) & (da["time"] < base_end))
    base_tmax = base_hourly.resample(time="1D").max()
    base_dates = _as_date_index(pd.to_datetime(base_tmax["time"].values))
    base_tmax = base_tmax.assign_coords(time=base_dates)
    base_tmax = base_tmax.sel(time=base_tmax["time"].dt.month.isin(months))

    # Per-station pctl threshold
    thr = base_tmax.quantile(args.pctl, dim="time", skipna=True)
    thr = thr.rename("tmax_threshold")

    # Detect in target year
    target_hourly = da.sel(time=(da["time"] >= year_start) & (da["time"] < year_end))
    target_tmax = target_hourly.resample(time="1D").max()
    target_dates = _as_date_index(pd.to_datetime(target_tmax["time"].values))
    target = target_tmax.assign_coords(time=target_dates)

    valid = np.isfinite(target) & np.isfinite(thr)
    exceed = (target > thr).where(valid)
    exceed_fraction = exceed.mean("weatherStation", skipna=True).astype(np.float32)

    frac = exceed_fraction.compute().values
    dates = _as_date_index(pd.to_datetime(exceed_fraction["time"].values))
    flagged = frac >= float(args.min_fraction)

    events = _events_from_flags(dates, flagged, frac, min_duration_days=int(args.min_duration))

    # Build event_id per day (0 = not in kept event)
    event_id = np.zeros_like(flagged, dtype=np.int32)
    for e in events:
        m = (dates >= e.start) & (dates <= e.end)
        event_id[m] = int(e.event_id)

    # Save thresholds
    thr_ds = xr.Dataset({"tmax_threshold": thr})
    thr_path = os.path.join(args.out_dir, "thresholds.nc")
    thr_ds.to_netcdf(thr_path)

    # Save daily series
    daily_df = pd.DataFrame(
        {
            "date": dates,
            "exceed_fraction": frac,
            "is_heatwave_day": flagged.astype(int),
            "event_id": event_id,
        }
    )
    daily_path = os.path.join(args.out_dir, f"daily_{args.target_year}.csv")
    daily_df.to_csv(daily_path, index=False)

    # Save events
    events_df = pd.DataFrame(
        [
            {
                "event_id": e.event_id,
                "start_date": e.start.date().isoformat(),
                "end_date": e.end.date().isoformat(),
                "duration_days": e.duration_days,
                "peak_exceed_fraction": e.peak_exceed_fraction,
                "mean_exceed_fraction": e.mean_exceed_fraction,
            }
            for e in events
        ]
    )
    events_path = os.path.join(args.out_dir, f"events_{args.target_year}.csv")
    events_df.to_csv(events_path, index=False)

    # Save day list used for events
    event_days = daily_df.loc[daily_df["event_id"] > 0, "date"].astype(str).tolist()
    days_path = os.path.join(args.out_dir, f"event_days_{args.target_year}.txt")
    with open(days_path, "w", encoding="utf-8") as f:
        for d in event_days:
            f.write(f"{d}\n")

    # Save hourly timestamps inside event days (for model eval sampling)
    time_index = pd.to_datetime(ds["time"].values)
    time_index = pd.to_datetime(time_index).tz_localize(None)
    if args.time_offset_hours:
        time_index = time_index + pd.to_timedelta(args.time_offset_hours, unit="h")
    time_dates = _as_date_index(time_index)
    event_day_set = set(pd.to_datetime(event_days))
    mask = np.array([d in event_day_set for d in time_dates], dtype=bool)
    times_path = os.path.join(args.out_dir, f"event_times_{args.target_year}.txt")
    with open(times_path, "w", encoding="utf-8") as f:
        for t in time_index[mask]:
            f.write(f"{t.isoformat()}\n")

    print("✅ AEMET-like heatwave derivation complete")
    print(f"   thresholds: {thr_path}")
    print(f"   daily:      {daily_path}")
    print(f"   events:     {events_path}")
    print(f"   days:       {days_path}")
    print(f"   times:      {times_path}")


if __name__ == "__main__":
    main()
