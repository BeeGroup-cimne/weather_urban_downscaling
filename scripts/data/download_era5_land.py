#!/usr/bin/env python3
"""
download_era5_land.py — Descarga ERA5-Land desde CDS API en formato GRIB.

Extraido de beemeteo/utils.py (BeeGroup-cimne/beemeteo, EUPL-1.2).
Sin dependencia HBase. Compacto, autocontenido.

Uso:
    python scripts/data/download_era5_land.py \
        --lat-range 41.0 42.5 \
        --lon-range 1.5 3.0 \
        --year-month 2017 201707 \
        --out-dir data/processed/era5land

Salida: data/processed/era5land/YYYYMM_lat_min_lon_max_lat_max.grib

Requiere:
    - cdsapi (pip install cdsapi)
    - credenciales CDS en ~/.cdsapirc
"""

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. Variables ERA5-Land
# ----------------------------------------------------------------------
ERA5LAND_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "forecast_albedo",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_4",
]


def _check_cds_credentials():
    rc_path = Path.home() / ".cdsapirc"
    if not rc_path.exists():
        print(
            "ERROR: No se encontraron credenciales CDS.\n"
            "  1. Registrate en https://cds.climate.copernicus.eu\n"
            "  2. Crea ~/.cdsapirc con:\n"
            "       url: https://cds.climate.copernicus.eu/api\n"
            "       key: <uid>:<api-key>\n"
        )
        sys.exit(1)


def _month_range_to_inclusive(year_month_start, year_month_end):
    """Convierte '2017' y '201707' a lista de ints YYYYMM inclusive.
    Si el valor es < 10000 se interpreta como solo YYYY (empieza en mes 1).
    Si es >= 10000 se interpreta como YYYYMM.
    """
    def _split(v):
        v = int(v)
        if v < 10000:
            return v, 1
        return divmod(v, 100)

    y, m = _split(year_month_start)
    end_y, end_m = _split(year_month_end)
    out = []
    while (y, m) <= (end_y, end_m):
        out.append(y * 100 + m)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _days_of_month(month_int):
    y = month_int // 100
    m = month_int % 100
    import calendar
    return [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def _hours_all():
    return [f"{h:02d}:00" for h in range(24)]


# ----------------------------------------------------------------------
# 2. Descarga CDS (basada en beemeteo get_hourly_historical_weather_from_ERA5Land)
# ----------------------------------------------------------------------
def download_era5_land_batch(
    lat_range,
    lon_range,
    year_months,
    out_dir,
    dry_run=False,
):
    """Descarga ERA5-Land por ano-mes via CDS API.

    Parametros
    ----------
    lat_range : tuple[float, float]
        (lat_min, lat_max)
    lon_range : tuple[float, float]
        (lon_min, lon_max)
    year_months : list[int]
        Ej: [201701, 201702, ...]
    out_dir : str
        Directorio donde guardar los GRIB.
    dry_run : bool
        True = solo imprimir lo que se descargaria.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not dry_run:
        _check_cds_credentials()
        import cdsapi
        c = cdsapi.Client()
    else:
        c = None

    area = [max(lat_range), min(lon_range), min(lat_range), max(lon_range)]

    for ym in year_months:
        year = ym // 100
        month = ym % 100
        fname = f"{ym}_{max(lat_range):.1f}_{min(lon_range):.1f}_"
        fname += f"{min(lat_range):.1f}_{max(lon_range):.1f}.grib"
        out_path = os.path.join(out_dir, fname)

        if os.path.exists(out_path):
            logger.info(f"SKIP {ym} -> {out_path} (exists)")
            continue

        logger.info(f"Descargando {ym} -> {out_path}")
        if dry_run:
            continue

        if c is None:
            continue
        c.retrieve(
            "reanalysis-era5-land",
            {
                "variable": ERA5LAND_VARIABLES,
                "year": f"{year}",
                "month": f"{month:02d}",
                "day": _days_of_month(ym),
                "time": _hours_all(),
                "area": area,
                "format": "grib",
            },
            out_path,
        )
        logger.info(f"OK {ym}")


# ----------------------------------------------------------------------
# 3. CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Descargar ERA5-Land historico desde CDS."
    )
    p.add_argument(
        "--lat-range", nargs=2, type=float, required=True,
        metavar=("LAT_MIN", "LAT_MAX"),
        help="Rango latitudinal (min max).",
    )
    p.add_argument(
        "--lon-range", nargs=2, type=float, required=True,
        metavar=("LON_MIN", "LON_MAX"),
        help="Rango longitudinal (min max).",
    )
    p.add_argument(
        "--year-month", nargs=2, required=True,
        metavar=("START", "END"),
        help="Rango YYYYMM o YYYY. Ej: 2017 201707",
    )
    p.add_argument(
        "--out-dir", default="data/processed/era5land",
        help="Directorio de salida (default: data/processed/era5land)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Solo mostrar que se descargaria, sin descargar.",
    )
    return p.parse_args(argv)


def main():
    args = parse_args()
    yms = _month_range_to_inclusive(args.year_month[0], args.year_month[1])
    print(f"Anos-meses a descargar: {yms}")
    print(f"Bounding box: lat={args.lat_range}, lon={args.lon_range}")
    print(f"Directorio: {args.out_dir}")
    if args.dry_run:
        print("--- DRY RUN ---")

    download_era5_land_batch(
        lat_range=tuple(args.lat_range),
        lon_range=tuple(args.lon_range),
        year_months=yms,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
