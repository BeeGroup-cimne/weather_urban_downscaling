import os
import sys
import numpy as np
import xarray as xr

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config.runtime import Config

# ================= CONFIG =================
RAW_DEFAULT = os.path.join(Config.BASE_DIR, "data", "static", "weather_static_raw.zarr")
ENGINEERED_DEFAULT = os.path.join(Config.BASE_DIR, "data", "static", "weather_static_engineered.zarr")
INPUT_PATH = os.environ.get("STATIC_INPUT_PATH", RAW_DEFAULT)
OUTPUT_PATH = os.environ.get("STATIC_OUTPUT_PATH", ENGINEERED_DEFAULT)

REQUIRED_RAW_VARS = [
    "building_area_residential_percentile",
    "n_dwellings_percentile",
    "building_area_industrial_percentile",
    "building_area_warehouse_parking_percentile",
    "building_area_offices_percentile",
    "building_area_commercial_percentile",
    "building_area_healthcare_and_charity_percentile",
    "building_area_religious_percentile",
    "building_area_cultural_percentile",
    "building_area_sports_facilities_percentile",
    "building_area_entertainment_venues_percentile",
    "building_area_leisure_and_hospitality_percentile",
    "building_area_singular_building_percentile",
    "n_floors_above_ground_percentile",
]

def engineer_static_features():
    print(f"Running static feature engineering from: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Static input not found: {INPUT_PATH}. "
            f"Set STATIC_INPUT_PATH to a valid raw static zarr."
        )

    ds = xr.open_zarr(INPUT_PATH)
    original_vars = set(str(v) for v in ds.data_vars.keys())
    missing_raw = [v for v in REQUIRED_RAW_VARS if v not in original_vars]

    if missing_raw:
        engineered_vars = [
            "elevation",
            "height_index",
            "ndvi_mean",
            "ndvi_min",
            "residential_index",
            "industrial_index",
            "services_index",
            "leisure_index",
        ]
        if set(engineered_vars).issubset(original_vars):
            print("Input already looks engineered; writing canonical copy.")
            ds_final = ds[engineered_vars].astype(np.float32)
            ds_final.attrs["static_engineering_version"] = "v1_already_engineered"
            ds_final.attrs["static_engineering_source"] = INPUT_PATH
            if "index" in ds_final.coords:
                ds_final.coords["index"] = ds_final.coords["index"].astype(str)
            if os.path.exists(OUTPUT_PATH):
                import shutil
                shutil.rmtree(OUTPUT_PATH)
            ds_final.to_zarr(OUTPUT_PATH, mode="w", consolidated=True)
            print(f"Wrote engineered static dataset to: {OUTPUT_PATH}")
            return
        raise RuntimeError(
            "Input dataset does not contain required raw static variables for engineering. "
            f"Missing: {missing_raw}"
        )

    print("Building composite static indices...")

    # A. INDICE RESIDENCIAL (Calor nocturno, calefacción)
    # Combina área residencial + número de viviendas
    ds['residential_index'] = (
        ds['building_area_residential_percentile'] + 
        ds['n_dwellings_percentile']
    ) / 2.0

    # B. INDICE INDUSTRIAL/LOGÍSTICO (Techos grandes, calor diurno)
    ds['industrial_index'] = (
        ds['building_area_industrial_percentile'] + 
        ds['building_area_warehouse_parking_percentile']
    ) / 2.0

    # C. INDICE SERVICIOS Y OFICINAS (Actividad diurna, AC)
    ds['services_index'] = (
        ds['building_area_offices_percentile'] + 
        ds['building_area_commercial_percentile'] + 
        ds['building_area_healthcare_and_charity_percentile'] + 
        ds['building_area_religious_percentile'] + 
        ds['building_area_cultural_percentile']
    ) / 5.0

    # D. INDICE OCIO Y GRANDES INSTALACIONES (Espacios singulares)
    ds['leisure_index'] = (
        ds['building_area_sports_facilities_percentile'] + 
        ds['building_area_entertainment_venues_percentile'] + 
        ds['building_area_leisure_and_hospitality_percentile'] + 
        ds['building_area_singular_building_percentile']
    ) / 4.0

    # Physical proxy
    ds['height_index'] = ds['n_floors_above_ground_percentile']

    # B. TOPOGRAFÍA
    # Mantenemos elevación tal cual (es la más importante)
    # ds['elevation'] ya existe

    # C. VEGETACIÓN (Refrigeración)
    # Mantenemos NDVI Mean y Min (para detectar agua/asfalto puro)
    # ndvi_max suele ser redundante con mean en ciudad
    # ds['ndvi_mean'] ya existe
    # ds['ndvi_min'] ya existe

    # Final keep list
    keep_vars = [
        'elevation',
        'height_index',
        'ndvi_mean',
        'ndvi_min',
        'residential_index',
        'industrial_index',
        'services_index',
        'leisure_index'
    ]

    ds_final = ds[keep_vars]
    ds_final = ds_final.astype(np.float32)

    print("Fixing coordinate dtypes...")
    if 'index' in ds_final.coords:
        ds_final.coords['index'] = ds_final.coords['index'].astype(str)
    for var in ds_final.coords:
        if 'chunks' in ds_final.coords[var].encoding:
            del ds_final.coords[var].encoding['chunks']

    ds_final.attrs["static_engineering_version"] = "v1"
    ds_final.attrs["static_engineering_source"] = INPUT_PATH

    if os.path.exists(OUTPUT_PATH):
        import shutil
        print(f"Removing previous output at {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)

    print(f"Writing engineered static zarr to: {OUTPUT_PATH}")
    ds_final.to_zarr(OUTPUT_PATH, mode='w', consolidated=True)
    print("Static feature engineering completed.")

if __name__ == "__main__":
    engineer_static_features()
