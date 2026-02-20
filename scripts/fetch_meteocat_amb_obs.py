#!/usr/bin/env python3
"""
Fetch real Meteocat station observations for strict AMB municipalities.

Outputs:
  1) Station metadata CSV
  2) Hourly observation CSV (long format) ready for Exp 2:
     columns: time, station_id, lat, lon, obs_c, n_obs_raw, ...
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE = "https://analisi.transparenciacatalunya.cat/resource"
DATASET_STATIONS = "yqwd-vj5e"
DATASET_VARIABLES = "4fb2-n3yi"
DATASET_OBS = "nzvn-apee"
PAGE_SIZE = 50000


AMB36_MUNICIPALITIES = {
    "badalona",
    "badia del valles",
    "barbera del valles",
    "barcelona",
    "begues",
    "castelldefels",
    "castellbisbal",
    "cerdanyola del valles",
    "cervello",
    "corbera de llobregat",
    "cornella de llobregat",
    "el papiol",
    "el prat de llobregat",
    "esplugues de llobregat",
    "gava",
    "l'hospitalet de llobregat",
    "la palma de cervello",
    "molins de rei",
    "montcada i reixac",
    "montgat",
    "palleja",
    "ripollet",
    "sant adria de besos",
    "sant andreu de la barca",
    "sant boi de llobregat",
    "sant climent de llobregat",
    "sant cugat del valles",
    "sant feliu de llobregat",
    "sant joan despi",
    "sant just desvern",
    "sant vicenc dels horts",
    "santa coloma de cervello",
    "santa coloma de gramenet",
    "tiana",
    "torrelles de llobregat",
    "viladecans",
}


def _norm_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text


def _fetch_json(dataset: str, params: dict):
    query = urlencode(params, safe="(),':")
    url = f"{BASE}/{dataset}.json?{query}"
    with urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_paginated(dataset: str, base_params: dict):
    offset = 0
    all_rows = []
    while True:
        params = dict(base_params)
        params["$limit"] = PAGE_SIZE
        params["$offset"] = offset
        rows = _fetch_json(dataset, params)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return all_rows


def _resolve_air_temperature_code() -> str:
    rows = _fetch_paginated(DATASET_VARIABLES, {})
    df = pd.DataFrame(rows)
    if "nom_variable" not in df.columns or "codi_variable" not in df.columns:
        raise RuntimeError("Could not resolve variable metadata from Meteocat dataset.")

    match = df[df["nom_variable"].str.contains("temperatura", case=False, na=False)].copy()
    if match.empty:
        raise RuntimeError("No 'temperatura' variable found in Meteocat metadata.")

    exact = match[match["nom_variable"].str.lower() == "temperatura de l'aire"]
    if not exact.empty:
        return str(exact.iloc[0]["codi_variable"])
    return str(match.iloc[0]["codi_variable"])


def _load_stations(include_only_operational: bool) -> pd.DataFrame:
    rows = _fetch_paginated(DATASET_STATIONS, {})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Station metadata query returned no rows.")

    df["mun_norm"] = df["nom_municipi"].map(_norm_text)
    df = df[df["mun_norm"].isin(AMB36_MUNICIPALITIES)].copy()
    if include_only_operational and "nom_estat_ema" in df.columns:
        df = df[df["nom_estat_ema"] == "Operativa"].copy()

    if df.empty:
        raise RuntimeError("No Meteocat stations found after AMB36 filtering.")

    for col in ["latitud", "longitud", "altitud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [
        "codi_estacio",
        "nom_estacio",
        "nom_municipi",
        "nom_comarca",
        "nom_provincia",
        "latitud",
        "longitud",
        "altitud",
        "nom_xarxa",
        "codi_estat_ema",
        "nom_estat_ema",
        "data_inici",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["codi_estacio"]).sort_values("codi_estacio").reset_index(drop=True)
    return df


def _fetch_obs_for_codes(codi_variable: str, station_codes: list, start: str, end: str) -> pd.DataFrame:
    if not station_codes:
        return pd.DataFrame(columns=["codi_estacio", "data_lectura", "valor_lectura"])

    all_frames = []
    chunk_size = 20
    for i in range(0, len(station_codes), chunk_size):
        chunk = station_codes[i : i + chunk_size]
        in_clause = ",".join([f"'{code}'" for code in chunk])
        where = (
            f"codi_variable='{codi_variable}' AND "
            f"codi_estacio IN ({in_clause}) AND "
            f"data_lectura between '{start}' and '{end}'"
        )
        rows = _fetch_paginated(
            DATASET_OBS,
            {
                "$select": "codi_estacio,data_lectura,valor_lectura",
                "$where": where,
                "$order": "data_lectura asc",
            },
        )
        if rows:
            all_frames.append(pd.DataFrame(rows))

    if not all_frames:
        return pd.DataFrame(columns=["codi_estacio", "data_lectura", "valor_lectura"])
    return pd.concat(all_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-01-01T00:00:00")
    parser.add_argument("--end", default="2017-12-31T23:59:59")
    parser.add_argument("--out-dir", default=os.path.join("experiments", "stations_eval", "data"))
    parser.add_argument("--only-operational", action="store_true", help="Keep currently operative stations only.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_code = _resolve_air_temperature_code()
    stations = _load_stations(include_only_operational=args.only_operational)

    station_out = out_dir / "meteocat_amb36_stations.csv"
    stations.to_csv(station_out, index=False)

    station_codes = stations["codi_estacio"].tolist()
    print(f"fetching observations for {len(station_codes)} stations...")
    obs = _fetch_obs_for_codes(temp_code, station_codes, args.start, args.end)

    if obs.empty:
        hourly = pd.DataFrame(columns=["time", "station_id", "lat", "lon", "obs_c", "n_obs_raw"])
    else:
        obs["data_lectura"] = pd.to_datetime(obs["data_lectura"], errors="coerce")
        obs["valor_lectura"] = pd.to_numeric(obs["valor_lectura"], errors="coerce")
        obs = obs.dropna(subset=["data_lectura", "valor_lectura", "codi_estacio"]).copy()
        obs["time"] = obs["data_lectura"].dt.floor("h")

        hourly = (
            obs.groupby(["time", "codi_estacio"], as_index=False)
            .agg(obs_c=("valor_lectura", "mean"), n_obs_raw=("valor_lectura", "size"))
            .rename(columns={"codi_estacio": "station_id"})
        )

        # If values look like Kelvin, convert to Celsius.
        if float(hourly["obs_c"].mean()) > 200.0:
            hourly["obs_c"] = hourly["obs_c"] - 273.15

        station_geo = (
            stations[["codi_estacio", "latitud", "longitud", "nom_estacio", "nom_municipi", "nom_comarca"]]
            .rename(columns={"codi_estacio": "station_id", "latitud": "lat", "longitud": "lon"})
        )
        hourly = hourly.merge(station_geo, on="station_id", how="left")
        hourly = hourly[
            [
                "time",
                "station_id",
                "nom_estacio",
                "nom_municipi",
                "nom_comarca",
                "lat",
                "lon",
                "obs_c",
                "n_obs_raw",
            ]
        ].sort_values(["time", "station_id"])

    obs_out = out_dir / "meteocat_amb36_air_temperature_hourly_2017.csv"
    hourly.to_csv(obs_out, index=False)

    coverage = (
        hourly.groupby("station_id", as_index=False)
        .agg(hours_with_data=("obs_c", "size"), mean_obs_c=("obs_c", "mean"))
        if not hourly.empty
        else pd.DataFrame(columns=["station_id", "hours_with_data", "mean_obs_c"])
    )
    coverage_out = out_dir / "meteocat_amb36_station_coverage_2017.csv"
    coverage.to_csv(coverage_out, index=False)

    print(f"temperature_variable_code={temp_code}")
    print(f"stations_selected={len(stations)}")
    print(f"obs_rows_hourly={len(hourly)}")
    print(f"saved={station_out}")
    print(f"saved={obs_out}")
    print(f"saved={coverage_out}")


if __name__ == "__main__":
    raise SystemExit(main())
