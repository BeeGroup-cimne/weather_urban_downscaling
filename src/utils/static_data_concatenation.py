import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path


def _bootstrap_proj_env() -> None:
    """Set PROJ env vars before importing geopandas/pyproj when possible."""
    existing_lib = os.getenv("PROJ_LIB", "").strip()
    existing_data = os.getenv("PROJ_DATA", "").strip()

    candidates = []
    candidates.append(os.path.join(sys.prefix, "share", "proj"))
    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "share", "proj"))
    candidates.extend([existing_lib, existing_data])
    candidates.extend(
        [
            "/opt/miniconda3/envs/ml_m4/share/proj",
            "/opt/miniconda3/share/proj",
        ]
    )
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "proj.db")):
            os.environ["PROJ_LIB"] = path
            os.environ["PROJ_DATA"] = path
            return


_bootstrap_proj_env()

import geopandas as gpd
from shapely.geometry import Point
from shapely.affinity import scale as scale_geom


RUTA_ESTACIONES_ZARR = "data/static/weather_static_engineered.zarr"
RUTA_OUTPUT = "data/processed/weather_static_FINAL_stations.zarr"

RADIO_BUFFER = 150  # meters
MIN_STREET_WIDTH_M = 6.0
OPEN_BLEND_LOW = 0.02
OPEN_BLEND_HIGH = 0.25
BBOX_PAD_FACTOR = 2.0


def _resolve_buildings_geojson() -> str:
    explicit = os.getenv("BUILDINGS_GEOJSON_PATH", "").strip()
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            "scripts/Scripts_Utils/barcelona_infalible.geojson",
            "src/utils/barcelona_infalible.geojson",
            "data/static/barcelona_infalible.geojson",
        ]
    )
    for p in candidates:
        if p and Path(p).exists():
            return p
    return candidates[0] if candidates else ""


def _stations_bbox_lonlat(df_estaciones: pd.DataFrame, pad_factor: float = BBOX_PAD_FACTOR):
    if df_estaciones.empty:
        return None
    lat_min = float(df_estaciones["lat"].min())
    lat_max = float(df_estaciones["lat"].max())
    lon_min = float(df_estaciones["lon"].min())
    lon_max = float(df_estaciones["lon"].max())

    lat0 = 0.5 * (lat_min + lat_max)
    pad_lat = pad_factor * (RADIO_BUFFER / 111_320.0)
    pad_lon = pad_factor * (RADIO_BUFFER / (111_320.0 * max(np.cos(np.deg2rad(lat0)), 1e-3)))
    return (lon_min - pad_lon, lat_min - pad_lat, lon_max + pad_lon, lat_max + pad_lat)


def _read_buildings_with_bbox(path: str, bbox):
    # bbox=(minx,miny,maxx,maxy) in lon/lat to avoid loading huge GeoJSON entirely.
    try:
        return gpd.read_file(path, bbox=bbox)
    except TypeError:
        gdf = gpd.read_file(path)
        if bbox is None:
            return gdf
        minx, miny, maxx, maxy = bbox
        return gdf.cx[minx:maxx, miny:maxy]


def _configure_proj_database() -> None:
    """Ensure pyproj can find proj.db in heterogeneous local environments."""
    try:
        import pyproj
        from pyproj import datadir as pyproj_datadir
    except Exception:
        return

    candidates = []
    env_proj_lib = os.getenv("PROJ_LIB", "").strip()
    if env_proj_lib:
        candidates.append(env_proj_lib)
    env_proj_data = os.getenv("PROJ_DATA", "").strip()
    if env_proj_data:
        candidates.append(env_proj_data)

    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "share", "proj"))
    candidates.append(os.path.join(sys.prefix, "share", "proj"))
    candidates.append("/opt/miniconda3/envs/ml_m4/share/proj")

    for path in candidates:
        if not path:
            continue
        proj_db = os.path.join(path, "proj.db")
        if os.path.exists(proj_db):
            try:
                pyproj_datadir.set_data_dir(path)
                os.environ["PROJ_LIB"] = path
                os.environ["PROJ_DATA"] = path
                return
            except Exception:
                continue


def _ensure_height_columns(gdf_edificios: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf_edificios.copy()
    if "height_m" not in out.columns:
        if "building:levels" in out.columns:
            levels = pd.to_numeric(out["building:levels"], errors="coerce").fillna(3.0)
        else:
            levels = pd.Series(3.0, index=out.index, dtype="float64")
        out["height_m"] = levels * 3.0
        if "levels_final" not in out.columns:
            out["levels_final"] = levels
    if "levels_final" not in out.columns:
        out["levels_final"] = np.maximum(1.0, np.round(out["height_m"] / 3.0))
    out["height_m"] = pd.to_numeric(out["height_m"], errors="coerce").fillna(0.0)
    out["levels_final"] = pd.to_numeric(out["levels_final"], errors="coerce").fillna(0.0)
    return out


def _sanitize_building_geometries(gdf_edificios: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf_edificios.copy()
    out = out[out.geometry.notnull()].copy()
    out = out[~out.geometry.is_empty].copy()

    poly_mask = out.geom_type.isin(["Polygon", "MultiPolygon"])
    if not poly_mask.all():
        out = out.loc[poly_mask].copy()

    if len(out):
        invalid_mask = ~out.geometry.is_valid
        if invalid_mask.any():
            out.loc[invalid_mask, "geometry"] = out.loc[invalid_mask, "geometry"].buffer(0)
            out = out[out.geometry.notnull() & ~out.geometry.is_empty].copy()
    return out


def _build_station_dataframe(indices: np.ndarray) -> pd.DataFrame:
    lat_lons = [str(x).split("_", 1) for x in indices]
    lats = [float(x[0]) for x in lat_lons]
    lons = [float(x[1]) for x in lat_lons]
    return pd.DataFrame({"index_zarr": indices, "lat": lats, "lon": lons})


def _compute_urban_metrics(
    gdf_buffers: gpd.GeoDataFrame,
    gdf_edificios: gpd.GeoDataFrame,
    area_buffer: float,
) -> pd.DataFrame:
    left = gdf_buffers[["index_zarr", "geometry"]].copy()
    right = gdf_edificios[["height_m", "levels_final", "geometry"]].copy()

    # True geometric intersection avoids over-counting whole buildings that only touch the buffer.
    inter = gpd.overlay(left, right, how="intersection", keep_geom_type=False)
    if inter.empty:
        return pd.DataFrame(
            index=gdf_buffers["index_zarr"],
            data={
                "total_built": 0.0,
                "avg_height": 0.0,
                "max_levels": 0.0,
                "buildings_in_buffer": 0,
            },
        )

    inter["inter_area_m2"] = inter.geometry.area
    inter = inter[inter["inter_area_m2"] > 0].copy()
    if inter.empty:
        return pd.DataFrame(
            index=gdf_buffers["index_zarr"],
            data={
                "total_built": 0.0,
                "avg_height": 0.0,
                "max_levels": 0.0,
                "buildings_in_buffer": 0,
            },
        )

    inter["height_x_area"] = inter["height_m"] * inter["inter_area_m2"]
    grouped = inter.groupby("index_zarr", dropna=False).agg(
        total_built=("inter_area_m2", "sum"),
        height_x_area=("height_x_area", "sum"),
        max_levels=("levels_final", "max"),
        buildings_in_buffer=("inter_area_m2", "size"),
    )
    grouped["avg_height"] = np.where(
        grouped["total_built"] > 0, grouped["height_x_area"] / grouped["total_built"], 0.0
    )
    grouped = grouped.drop(columns=["height_x_area"])

    # Clamp impossible over-coverage due to geometry artifacts.
    grouped["total_built"] = grouped["total_built"].clip(lower=0.0, upper=area_buffer)
    return grouped


def _derive_v2_from_existing_output(output_path: str) -> xr.Dataset:
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Fallback requires existing output dataset at {output_path}, but it does not exist."
        )
    ds_prev = xr.open_zarr(output_path, consolidated=True)
    required = ["avg_height", "building_density", "svf"]
    missing = [v for v in required if v not in ds_prev.data_vars]
    if missing:
        raise RuntimeError(
            f"Fallback dataset missing required vars {missing} in {output_path}."
        )

    avg_height = ds_prev["avg_height"].astype(np.float32)
    dens = ds_prev["building_density"].clip(0.0, 1.0).astype(np.float32)
    svf = ds_prev["svf"].clip(0.0, 1.0).astype(np.float32)

    if np.all(~np.isfinite(avg_height.values)) or np.all(~np.isfinite(dens.values)) or np.all(~np.isfinite(svf.values)):
        raise RuntimeError("Existing output static dataset is fully NaN in required fields.")

    # Invert canyon relation approximately: svf ~= cos(arctan(2h/w))
    eps = 1e-4
    h_over_w = (0.5 * np.tan(np.arccos(svf.clip(eps, 1.0 - eps)))).astype(np.float32)
    street_width_m = (avg_height / (h_over_w + eps)).clip(MIN_STREET_WIDTH_M, 2.0 * RADIO_BUFFER).astype(np.float32)
    svf_canyon = np.cos(np.arctan(2.0 * h_over_w)).clip(0.0, 1.0).astype(np.float32)

    if "ndvi_mean" in ds_prev.data_vars:
        ndvi = ds_prev["ndvi_mean"].clip(0.0, 1.0).astype(np.float32)
        impervious_fraction = (0.65 * dens + 0.35 * (1.0 - ndvi)).clip(0.0, 1.0).astype(np.float32)
    else:
        impervious_fraction = dens

    out = ds_prev.copy()
    out["street_width_m"] = street_width_m
    out["h_over_w"] = h_over_w
    out["svf_canyon"] = svf_canyon
    out["impervious_fraction"] = impervious_fraction
    if "buildings_in_buffer" not in out.data_vars:
        out["buildings_in_buffer"] = xr.zeros_like(dens, dtype=np.float32)

    out.attrs["static_schema_version"] = "svf_v2_morphology_v1_fallback_from_existing"
    out.attrs["svf_method"] = "canyon_continuous_blend_v2_fallback"
    out.attrs["svf_buffer_radius_m"] = float(RADIO_BUFFER)
    out.attrs["svf_blend_density_low"] = float(OPEN_BLEND_LOW)
    out.attrs["svf_blend_density_high"] = float(OPEN_BLEND_HIGH)
    out.attrs["impervious_method"] = "0.65*building_density + 0.35*(1-ndvi_mean)"
    return out


def _derive_v2_from_engineered_base(ds_base: xr.Dataset) -> xr.Dataset:
    required = ["height_index", "ndvi_mean", "residential_index", "industrial_index", "services_index"]
    missing = [v for v in required if v not in ds_base.data_vars]
    if missing:
        raise RuntimeError(f"Engineered base dataset missing required vars for synthetic fallback: {missing}")

    out = ds_base.copy()
    h_idx = out["height_index"].clip(0.0, 1.0).astype(np.float32)
    ndvi = out["ndvi_mean"].clip(0.0, 1.0).astype(np.float32)
    resi = out["residential_index"].clip(0.0, 1.0).astype(np.float32)
    ind = out["industrial_index"].clip(0.0, 1.0).astype(np.float32)
    serv = out["services_index"].clip(0.0, 1.0).astype(np.float32)

    dens = (0.55 * resi + 0.25 * serv + 0.20 * ind).clip(0.0, 1.0).astype(np.float32)
    avg_height = (2.5 + 32.0 * h_idx).astype(np.float32)
    max_levels = np.round(avg_height / 3.0).astype(np.float32)
    street_width_m = (2.0 * RADIO_BUFFER * np.sqrt((1.0 - dens).clip(1e-4, 1.0))).clip(
        MIN_STREET_WIDTH_M, 2.0 * RADIO_BUFFER
    ).astype(np.float32)
    h_over_w = (avg_height / (street_width_m + 1e-4)).astype(np.float32)
    svf_canyon = np.cos(np.arctan(2.0 * h_over_w)).clip(0.0, 1.0).astype(np.float32)
    blend = ((dens - OPEN_BLEND_LOW) / (OPEN_BLEND_HIGH - OPEN_BLEND_LOW)).clip(0.0, 1.0).astype(np.float32)
    svf = ((1.0 - blend) + blend * svf_canyon).clip(0.0, 1.0).astype(np.float32)
    roughness = (0.15 * avg_height * (1.0 - np.exp(-np.sqrt(4.0 * dens)))).clip(min=0.001).astype(np.float32)
    impervious = (0.65 * dens + 0.35 * (1.0 - ndvi)).clip(0.0, 1.0).astype(np.float32)
    buildings_in_buffer = np.round(dens * 220.0).astype(np.float32)

    out["avg_height"] = avg_height
    out["building_density"] = dens
    out["max_levels"] = max_levels
    out["street_width_m"] = street_width_m
    out["h_over_w"] = h_over_w
    out["svf_canyon"] = svf_canyon
    out["svf"] = svf
    out["roughness"] = roughness
    out["impervious_fraction"] = impervious
    out["buildings_in_buffer"] = buildings_in_buffer

    out.attrs["static_schema_version"] = "svf_v2_morphology_v1_fallback_synthetic"
    out.attrs["svf_method"] = "synthetic_from_engineered_base_v1"
    out.attrs["impervious_method"] = "0.65*synthetic_density + 0.35*(1-ndvi_mean)"
    return out


def enriquecer_estaciones():
    _configure_proj_database()
    print("Starting static enrichment...")
    ds = xr.open_zarr(RUTA_ESTACIONES_ZARR, consolidated=True)
    indices = ds.index.values
    print(f"  Points: {len(indices):,}")

    df_estaciones = _build_station_dataframe(indices)
    station_geoms = [Point(xy) for xy in zip(df_estaciones["lon"], df_estaciones["lat"])]
    try:
        gdf_estaciones = gpd.GeoDataFrame(
            df_estaciones,
            geometry=station_geoms,
            crs="EPSG:4326",
        )
    except Exception as e:
        print(f"  ⚠️ CRS init unavailable ({e}). Building station GeoDataFrame without CRS.")
        gdf_estaciones = gpd.GeoDataFrame(
            df_estaciones,
            geometry=station_geoms,
        )
        gdf_estaciones = gdf_estaciones.set_crs(None, allow_override=True)

    print("  Loading building footprints...")
    ruta_edificios = _resolve_buildings_geojson()
    if not os.path.exists(ruta_edificios):
        print(f"  ⚠️ Building GeoJSON not found (tried default locations). Using fallback static derivation.")
        try:
            ds_fallback = _derive_v2_from_existing_output(RUTA_OUTPUT)
            print("  ℹ️ Fallback source: existing output static dataset.")
        except Exception:
            ds_fallback = _derive_v2_from_engineered_base(ds)
            print("  ℹ️ Fallback source: engineered base static dataset (synthetic morphology).")
        ds_fallback.to_zarr(RUTA_OUTPUT, mode="w", consolidated=True)
        print("Fallback enrichment finished.")
        return ds_fallback.to_dataframe()

    print(f"  Using buildings file: {ruta_edificios}")
    bbox = _stations_bbox_lonlat(df_estaciones)
    if bbox is not None:
        print(
            "  Reading buildings with bbox "
            f"(minx={bbox[0]:.5f}, miny={bbox[1]:.5f}, maxx={bbox[2]:.5f}, maxy={bbox[3]:.5f})"
        )
    gdf_edificios = _read_buildings_with_bbox(ruta_edificios, bbox)
    print(f"  Buildings loaded after bbox filter: {len(gdf_edificios):,}")
    if gdf_edificios.empty:
        raise RuntimeError(
            "Building GeoJSON loaded but returned 0 features in station bbox. "
            "Check BUILDINGS_GEOJSON_PATH or bbox assumptions."
        )
    before_sanitize = len(gdf_edificios)
    gdf_edificios = _sanitize_building_geometries(gdf_edificios)
    print(f"  Buildings usable polygons: {len(gdf_edificios):,} / {before_sanitize:,}")
    if gdf_edificios.empty:
        raise RuntimeError("No valid polygon building geometries after sanitation.")
    gdf_edificios = _ensure_height_columns(gdf_edificios)

    gdf_est_metric = None
    gdf_bld_metric = None
    try:
        print("  Reprojecting to metric CRS...")
        gdf_est_metric = gdf_estaciones.to_crs("EPSG:25831")
        gdf_bld_metric = gdf_edificios.to_crs("EPSG:25831")
    except Exception as e:
        print(f"  ⚠️ CRS reprojection unavailable ({e}). Using local metric approximation fallback.")
        # Fallback: local equirectangular scaling around Barcelona latitude.
        lat0 = float(df_estaciones["lat"].mean()) if len(df_estaciones) else 41.39
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))

        def _to_local_metric(geom):
            if geom is None or getattr(geom, "is_empty", False):
                return geom
            return scale_geom(geom, xfact=m_per_deg_lon, yfact=m_per_deg_lat, origin=(0.0, 0.0))

        gdf_est_metric = gdf_estaciones.copy()
        gdf_est_metric["geometry"] = gdf_est_metric.geometry.apply(_to_local_metric)
        gdf_est_metric = gdf_est_metric.set_crs(None, allow_override=True)

        gdf_bld_metric = gdf_edificios.copy()
        gdf_bld_metric["geometry"] = gdf_bld_metric.geometry.apply(_to_local_metric)
        gdf_bld_metric = gdf_bld_metric.set_crs(None, allow_override=True)

    print(f"  Building influence buffers: {RADIO_BUFFER} m")
    gdf_buffers = gdf_est_metric[["index_zarr", "geometry"]].copy()
    gdf_buffers["geometry"] = gdf_buffers.geometry.buffer(RADIO_BUFFER)
    area_buffer = float(np.pi * (RADIO_BUFFER**2))

    print("  Computing geometric intersections (area-weighted metrics)...")
    urban_stats = _compute_urban_metrics(gdf_buffers, gdf_bld_metric, area_buffer=area_buffer)

    df_final = df_estaciones.set_index("index_zarr").join(urban_stats).fillna(0.0)

    print("  Deriving morphology features...")
    df_final["building_density"] = (df_final["total_built"] / area_buffer).clip(lower=0.0, upper=1.0)
    open_fraction = (1.0 - df_final["building_density"]).clip(lower=1e-4, upper=1.0)
    df_final["street_width_m"] = (2.0 * RADIO_BUFFER * np.sqrt(open_fraction)).clip(
        lower=MIN_STREET_WIDTH_M, upper=2.0 * RADIO_BUFFER
    )
    df_final["h_over_w"] = np.where(
        df_final["street_width_m"] > 0, df_final["avg_height"] / df_final["street_width_m"], 0.0
    )

    # Canyon SVF model with continuous blending to open terrain.
    svf_canyon = np.cos(np.arctan(2.0 * df_final["h_over_w"]))
    blend = ((df_final["building_density"] - OPEN_BLEND_LOW) / (OPEN_BLEND_HIGH - OPEN_BLEND_LOW)).clip(0.0, 1.0)
    df_final["svf"] = ((1.0 - blend) * 1.0 + blend * svf_canyon).clip(0.0, 1.0)
    df_final["svf_canyon"] = svf_canyon.clip(0.0, 1.0)

    # Roughness proxy (bounded, continuous with density and height).
    dens = df_final["building_density"].clip(lower=0.0, upper=1.0)
    df_final["roughness"] = (
        0.15 * df_final["avg_height"] * (1.0 - np.exp(-np.sqrt(4.0 * dens)))
    ).clip(lower=0.001)

    # Impervious proxy combines built fraction and lack of vegetation.
    if "ndvi_mean" in df_final.columns:
        ndvi = df_final["ndvi_mean"].clip(0.0, 1.0)
        df_final["impervious_fraction"] = (0.65 * dens + 0.35 * (1.0 - ndvi)).clip(0.0, 1.0)
    else:
        df_final["impervious_fraction"] = dens

    print(f"  Writing output: {RUTA_OUTPUT}")
    enrich_cols = [
        "avg_height",
        "building_density",
        "svf",
        "roughness",
        "max_levels",
        "street_width_m",
        "h_over_w",
        "svf_canyon",
        "impervious_fraction",
        "buildings_in_buffer",
    ]
    ds_vars = xr.Dataset.from_dataframe(df_final[enrich_cols]).rename({"index_zarr": "index"})
    ds_final = xr.merge([ds, ds_vars], compat="override")
    ds_final.attrs["static_schema_version"] = "svf_v2_morphology_v1"
    ds_final.attrs["svf_method"] = "canyon_continuous_blend_v2"
    ds_final.attrs["svf_buffer_radius_m"] = float(RADIO_BUFFER)
    ds_final.attrs["svf_blend_density_low"] = float(OPEN_BLEND_LOW)
    ds_final.attrs["svf_blend_density_high"] = float(OPEN_BLEND_HIGH)
    ds_final.attrs["impervious_method"] = "0.65*building_density + 0.35*(1-ndvi_mean)"
    ds_final.attrs["static_feature_order_legacy"] = json.dumps(
        [
            "avg_height",
            "building_density",
            "elevation",
            "height_index",
            "industrial_index",
            "leisure_index",
            "max_levels",
            "ndvi_mean",
            "ndvi_min",
            "residential_index",
            "roughness",
            "services_index",
            "svf",
        ],
        ensure_ascii=True,
    )
    ds_final.attrs["static_feature_order_v2"] = json.dumps(
        [
            "avg_height",
            "building_density",
            "impervious_fraction",
            "elevation",
            "height_index",
            "industrial_index",
            "leisure_index",
            "ndvi_mean",
            "ndvi_min",
            "residential_index",
            "services_index",
            "svf",
            "h_over_w",
        ],
        ensure_ascii=True,
    )
    ds_final.to_zarr(RUTA_OUTPUT, mode="w", consolidated=True)

    print("Static enrichment finished.")
    return df_final


def visualizar_validacion(df):
    if os.getenv("STATIC_QUICKLOOK", "0").strip().lower() not in {"1", "true", "yes"}:
        return
    if not {"lon", "lat"}.issubset(set(df.columns)):
        print("Skipping SVF quicklook: dataframe has no lon/lat columns.")
        return
    print("Generating SVF quicklook...")
    plt.figure(figsize=(10, 8))
    plt.scatter(df.lon, df.lat, c=df.svf, cmap="magma", s=5, alpha=0.7)
    plt.colorbar(label="Sky View Factor (0=closed, 1=open)")
    plt.title("SVF validation map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


if __name__ == "__main__":
    df_res = enriquecer_estaciones()
    visualizar_validacion(df_res)
