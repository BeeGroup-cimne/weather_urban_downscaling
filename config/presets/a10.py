"""
Preset recomendado para servidor NVIDIA A10 (22GB).
Copiar estos valores a config/gpu_server_config.py según necesidad.
"""

PRESET = {
    "BATCH_SIZE": 2,
    "GRADIENT_ACCUMULATION_STEPS": 2,
    "SEQ_LEN": 6,
    "MIXED_PRECISION": False,
    "TEMPORAL_STRIDE": 2,
    "TEMPORAL_SAMPLER": "weighted",
    "TEMPORAL_WEIGHT_GAMMA": 1.0,
    "TEMPORAL_MIN_PROB": 1e-6,
    "TEMPORAL_SEASON_BALANCE": True,
    "STATION_GRIB_PATH": "data/processed/stations_t2m.grib",
}
