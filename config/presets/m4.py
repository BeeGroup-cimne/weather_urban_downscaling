"""
Preset recomendado para Apple M4 (MPS).
Copiar estos valores a config/config.py según necesidad.
"""

PRESET = {
    "BATCH_SIZE": 2,
    "SEQ_LEN": 6,
    "MAX_STEPS_PER_EPOCH": 200,
    "TEMPORAL_STRIDE": 2,
    "TEMPORAL_SAMPLER": "weighted",
    "TEMPORAL_WEIGHT_GAMMA": 1.0,
    "TEMPORAL_MIN_PROB": 1e-6,
    "TEMPORAL_SEASON_BALANCE": True,
    "STATION_GRIB_PATH": "data/processed/stations_t2m.grib",
}
