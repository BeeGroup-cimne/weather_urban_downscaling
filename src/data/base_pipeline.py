"""
Base data pipeline — shared temporal-sampling and utility methods.

Extracted from the common logic duplicated across:
  - BigDataPipeline       (src/data_loader.py)
  - TileDataPipeline      (src/data_loader_tiles.py)
  - OptimizedBigDataPipeline (src/optimized_data_pipeline.py)

Subclasses override `get_tf_datasets()` and `process_static_data()` but inherit
all temporal helpers without re-implementing them.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr


class TemporalSamplerMixin:
    """Mixin providing reusable temporal-sampling primitives.

    Subclasses must set:
      - self.cfg   (a Config-like object with TEMPORAL_* / SEASON_* attrs)
      - self._rng  (numpy Generator instance)
    """

    # ------------------------------------------------------------------
    # Pure helpers (no side-effects, easily testable)
    # ------------------------------------------------------------------

    @staticmethod
    def order_values(vals: np.ndarray) -> str:
        """Return 'asc', 'desc', or 'unknown' for a 1-D coordinate array."""
        diffs = np.diff(vals)
        if np.all(diffs > 0):
            return "asc"
        if np.all(diffs < 0):
            return "desc"
        return "unknown"

    @staticmethod
    def time_indices(times, start, end):
        """Convert date strings to integer indices via searchsorted.

        Args:
            times: Array-like of datetime values.
            start: Start date string (e.g. '2017-01-01').
            end:   End date string.

        Returns:
            Tuple[int, int]: (start_idx, end_idx).
        """
        times = pd.to_datetime(times).values
        start = np.datetime64(start)
        end = np.datetime64(end)
        start_idx = int(np.searchsorted(times, start, side="left"))
        end_idx = int(np.searchsorted(times, end, side="left"))
        return start_idx, end_idx

    @staticmethod
    def season_index(times_idx) -> np.ndarray:
        """Map a DatetimeIndex to season IDs: DJF=0, MAM=1, JJA=2, SON=3."""
        months = times_idx.month
        seasons = np.zeros_like(months)
        seasons[(months >= 3) & (months <= 5)] = 1
        seasons[(months >= 6) & (months <= 8)] = 2
        seasons[(months >= 9) & (months <= 11)] = 3
        return seasons

    # ------------------------------------------------------------------
    # Stateful helpers (rely on self.cfg and self._rng)
    # ------------------------------------------------------------------

    def sample_time(
        self,
        start_i: int,
        end_i: int,
        max_start: int,
        *,
        temporal_sampler: str,
        season_balance: bool,
        seasons: np.ndarray,
        time_weights: np.ndarray | None,
    ) -> int:
        """Draw a start index for a temporal window.

        Supports uniform, weighted, and season-balanced sampling strategies.

        Args:
            start_i:          Minimum valid index.
            end_i:            Maximum valid index (exclusive).
            max_start:        Upper bound for the start index.
            temporal_sampler: 'uniform' | 'weighted' | 'weighted_station'.
            season_balance:   Whether to balance across seasons.
            seasons:          Array of season IDs (from ``season_index``).
            time_weights:     Optional probability weights per timestep.

        Returns:
            int: Chosen start index.
        """
        rng = self._rng
        if max_start is None or max_start <= start_i:
            return start_i
        if temporal_sampler == "uniform" and not season_balance:
            return int(rng.integers(start_i, max_start))

        candidates = np.arange(start_i, max_start)
        if candidates.size == 0:
            return int(start_i)

        if season_balance:
            season_ids = [0, 1, 2, 3]
            available = [s for s in season_ids if np.any(seasons[candidates] == s)]
            if available:
                chosen_season = rng.choice(available)
                candidates = candidates[seasons[candidates] == chosen_season]

        if time_weights is None:
            return int(rng.choice(candidates))

        weights = time_weights[candidates]
        weights = weights / np.sum(weights)
        return int(rng.choice(candidates, p=weights))

    def build_time_weights(self, da_hr, ds, *, temporal_sampler: str) -> np.ndarray | None:
        """Compute per-timestep sampling weights from temperature gradients.

        Supports two modes:
          - ``weighted``: weights derived from the HR spatial-mean series.
          - ``weighted_station``: weights derived from an external station GRIB
            (falls back to ``weighted`` on failure).

        Args:
            da_hr:            HR DataArray (with 'time' dim).
            ds:               Full xarray Dataset (for time coordinate alignment).
            temporal_sampler: Sampling strategy string.

        Returns:
            1-D float32 array of probabilities, or None if uniform.
        """
        if temporal_sampler not in ("weighted", "weighted_station"):
            return None

        series = None
        if temporal_sampler == "weighted_station":
            series = self._load_station_series(ds)

        if series is None:
            try:
                hr_mean = da_hr.mean(dim=[d for d in da_hr.dims if d != "time"]).values
                series = hr_mean
            except Exception:
                return None

        if series is None:
            return None

        series = np.asarray(series, dtype=np.float32)
        grad = np.abs(np.diff(series, prepend=series[0]))
        grad = grad - np.nanmin(grad)
        gamma = float(getattr(self.cfg, "TEMPORAL_WEIGHT_GAMMA", 1.0))
        if gamma != 1.0:
            grad = np.power(grad, gamma)
        min_prob = float(getattr(self.cfg, "TEMPORAL_MIN_PROB", 1e-6))
        grad = grad + min_prob
        grad = grad / np.sum(grad)
        return grad

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _load_station_series(self, ds) -> np.ndarray | None:
        """Try to load a temperature series from the station GRIB."""
        station_path = getattr(self.cfg, "STATION_GRIB_PATH", "")
        if not station_path or not os.path.exists(station_path):
            return None
        try:
            cfgrib_kwargs = {
                "filter_by_keys": {"typeOfLevel": "surface"},
                "errors": "ignore",
                "indexpath": "",
            }
            ds_st = xr.open_dataset(station_path, engine="cfgrib", backend_kwargs=cfgrib_kwargs)
            var = None
            for v in ["t2m", "2t", "tas", "airTemperature"]:
                if v in ds_st:
                    var = v
                    break
            if var is None:
                var = list(ds_st.data_vars)[0]
            da = ds_st[var]
            if float(da.isel({da.dims[0]: slice(0, min(3, da.sizes[da.dims[0]]))}).mean().values) > 200:
                da = da - 273.15
            time_dim = next((d for d in da.dims if d in ["time", "valid_time"]), da.dims[0])
            reduce_dims = [d for d in da.dims if d != time_dim]
            series = da.mean(dim=reduce_dims).values
            station_times = pd.to_datetime(da[time_dim].values).floor("H")
            ds_times = pd.to_datetime(ds["time"].values).floor("H")
            time_map = {t: i for i, t in enumerate(station_times)}
            aligned = np.zeros(ds_times.shape[0], dtype=np.float32)
            for i, t in enumerate(ds_times):
                j = time_map.get(t)
                if j is not None:
                    aligned[i] = series[j]
            return aligned
        except Exception:
            print("⚠️ No se pudo usar estaciones para pesos temporales, fallback a HR.")
            return None
