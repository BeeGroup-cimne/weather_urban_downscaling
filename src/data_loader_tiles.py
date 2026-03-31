# Tile-based data pipeline (independent from the full-frame pipeline).

import os
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

from config.runtime import Config
from src.data_loader import BigDataPipeline
from src.data.base_pipeline import TemporalSamplerMixin
from src.utils.static_features import read_static_cache_meta, robust_unit_scale, select_static_feature_names


class TileDataPipeline:
    def __init__(self, config=None):
        self.cfg = config or Config
        self.cache_dir = self.cfg.PATH_CACHE
        self.static_cache_path = self.cfg.STATIC_CACHE_PATH
        self.static_norm = None
        self.static_raw = None
        self.static_var_names = None
        self.weight_map = None
        self.weight_cdf = None
        self.time_weight = None
        self.time_cdf = None
        self.time_series = None
        self.rng = np.random.default_rng(getattr(self.cfg, "SEED", 42))

    def _ensure_cache(self):
        if os.path.exists(self.cache_dir) and os.path.exists(self.static_cache_path):
            return
        pipeline = BigDataPipeline(self.cfg)
        pipeline.process_static_data()
        pipeline.run_etl_process()

    def _load_static(self, hr_shape):
        if not os.path.exists(self.static_cache_path):
            pipeline = BigDataPipeline(self.cfg)
            pipeline.process_static_data()
        static = np.load(self.static_cache_path)
        if static.ndim == 2:
            static = static[..., np.newaxis]
        if static.shape[0] != hr_shape[0] or static.shape[1] != hr_shape[1]:
            static = tf.image.resize(static, hr_shape, method="bilinear").numpy()
        self.static_raw = static.astype(np.float32)
        mean_st = np.mean(static, axis=(0, 1), keepdims=True)
        std_st = np.std(static, axis=(0, 1), keepdims=True)
        self.static_norm = (static - mean_st) / (std_st + 1e-6)
        if np.isnan(self.static_norm).any():
            self.static_norm = np.nan_to_num(self.static_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Prefer metadata written with static cache; fallback to schema selection from zarr.
        meta = read_static_cache_meta(self.static_cache_path)
        if meta and isinstance(meta.get("feature_names"), list):
            self.static_var_names = [str(v) for v in meta["feature_names"]]
        else:
            try:
                ds_static = xr.open_zarr(self.cfg.PATH_STATIC, consolidated=True)
                selected, _ = select_static_feature_names(ds_static, self.cfg)
                requested = [str(v) for v in getattr(self.cfg, "STATIC_FEATURES", selected)]
                self.static_var_names = requested if requested else selected
            except Exception:
                self.static_var_names = None
        return self.static_norm

    def _build_weight_map(self, sampler="static_weighted", hr_shape=None):
        if sampler == "static_weighted":
            static = self.static_norm
            # Importance by mean absolute normalized static signal
            weight = np.mean(np.abs(static), axis=-1)
        elif sampler == "uhi_proxy":
            # Heatwave/UHI-oriented proxy:
            # emphasize built-up / roughness, de-emphasize vegetation (cooling).
            # Uses physical-scale static channels when names are available.
            static = self.static_raw if self.static_raw is not None else self.static_norm
            names = self.static_var_names or []

            def _idx(name: str):
                try:
                    return names.index(name)
                except Exception:
                    return None

            idx_bd = _idx("building_density")
            idx_ndvi = _idx("ndvi_mean")
            idx_h = _idx("avg_height") if _idx("avg_height") is not None else _idx("height_index")
            idx_r = _idx("roughness")
            idx_svf = _idx("svf")

            idx_imp = _idx("impervious_fraction")
            idx_hw = _idx("h_over_w")

            if idx_bd is None and idx_imp is None:
                # Not enough info; default to generic static_weighted
                weight = np.mean(np.abs(static), axis=-1)
            else:
                built = static[:, :, idx_imp] if idx_imp is not None else static[:, :, idx_bd]
                ndvi = static[:, :, idx_ndvi] if idx_ndvi is not None else np.zeros_like(built)
                h = static[:, :, idx_h] if idx_h is not None else np.zeros_like(built)
                r = static[:, :, idx_r] if idx_r is not None else np.zeros_like(built)
                svf = static[:, :, idx_svf] if idx_svf is not None else np.ones_like(built)
                hw = static[:, :, idx_hw] if idx_hw is not None else np.zeros_like(built)

                built_n = robust_unit_scale(built)
                ndvi_n = robust_unit_scale(ndvi)
                h_n = robust_unit_scale(h)
                r_n = robust_unit_scale(r)
                canyon_n = robust_unit_scale((1.0 - np.clip(svf, 0.0, 1.0)) + hw)

                score = (
                    0.45 * built_n
                    + 0.25 * canyon_n
                    + 0.15 * h_n
                    + 0.10 * r_n
                    + 0.05 * (1.0 - ndvi_n)
                )
                weight = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        elif sampler == "error_weighted":
            err_path = getattr(self.cfg, "TILE_ERROR_MAP_PATH", "")
            if not err_path or not os.path.exists(err_path):
                print("⚠️ Error-weighted sampler requested but no error map found. Falling back to static_weighted.")
                return self._build_weight_map("static_weighted", hr_shape=hr_shape)
            weight = np.load(err_path)
            if hr_shape and (weight.shape[0] != hr_shape[0] or weight.shape[1] != hr_shape[1]):
                weight = tf.image.resize(weight[..., None], hr_shape, method="bilinear").numpy()[..., 0]
        else:
            return None

        weight = weight - np.nanmin(weight)
        gamma = float(getattr(self.cfg, "TILE_WEIGHT_GAMMA", 1.0))
        if gamma != 1.0:
            weight = np.power(weight, gamma)
        min_prob = float(getattr(self.cfg, "TILE_MIN_PROB", 1e-6))
        weight = weight + min_prob
        weight = weight / np.sum(weight)
        self.weight_map = weight
        self.weight_cdf = np.cumsum(weight.ravel())
        return weight

    def _sample_patch_top_left(self, hr_h, hr_w, patch_h, patch_w, sampler="static_weighted"):
        weighted_samplers = ("static_weighted", "error_weighted", "uhi_proxy")
        if sampler not in weighted_samplers or self.weight_cdf is None:
            y0 = self.rng.integers(0, max(1, hr_h - patch_h + 1))
            x0 = self.rng.integers(0, max(1, hr_w - patch_w + 1))
            return int(y0), int(x0)

        # Sample a center with importance weights, then clamp
        r = self.rng.random()
        idx = int(np.searchsorted(self.weight_cdf, r, side="right"))
        idx = min(idx, self.weight_cdf.size - 1)
        cy = idx // hr_w
        cx = idx % hr_w
        y0 = int(cy - patch_h // 2)
        x0 = int(cx - patch_w // 2)
        y0 = max(0, min(y0, hr_h - patch_h))
        x0 = max(0, min(x0, hr_w - patch_w))
        return y0, x0

    def _build_time_weights(self, da_hr, time_dim):
        sampler = getattr(self.cfg, "TEMPORAL_SAMPLER", "uniform")
        self.time_weight = None
        self.time_cdf = None
        self.time_series = None
        if sampler not in ("weighted", "weighted_station", "p95"):
            return None

        series = None

        if sampler == "weighted_station":
            station_path = getattr(self.cfg, "STATION_GRIB_PATH", "")
            if station_path and os.path.exists(station_path):
                try:
                    cfgrib_kwargs = {
                        "filter_by_keys": {"typeOfLevel": "surface"},
                        "errors": "ignore",
                        "indexpath": "",
                    }
                    ds_st = xr.open_dataset(station_path, engine="cfgrib", backend_kwargs=cfgrib_kwargs)
                    for v in ["t2m", "2t", "tas", "airTemperature"]:
                        if v in ds_st:
                            var = v
                            break
                    else:
                        var = list(ds_st.data_vars)[0]
                    da = ds_st[var]
                    if float(da.isel({da.dims[0]: slice(0, min(3, da.sizes[da.dims[0]]))}).mean().values) > 200:
                        da = da - 273.15
                    time_d = next((d for d in da.dims if d in ["time", "valid_time"]), da.dims[0])
                    reduce_dims = [d for d in da.dims if d != time_d]
                    series = da.mean(dim=reduce_dims).values
                except Exception:
                    series = None

        if series is None:
            try:
                hr_mean = da_hr.mean(dim=[d for d in da_hr.dims if d != time_dim]).compute()
                series = hr_mean.values
            except Exception:
                print("⚠️ No se pudo construir pesos temporales, fallback uniform.")
                self.time_cdf = None
                return None

        series = np.asarray(series, dtype=np.float32)
        if series.size == 0 or not np.isfinite(series).any():
            print("⚠️ Serie temporal inválida para pesos, fallback uniform.")
            return None

        if sampler == "p95":
            self.time_series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            return None

        grad = np.abs(np.diff(series, prepend=series[0]))
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        w = grad

        w = w - np.nanmin(w)
        gamma = float(getattr(self.cfg, "TEMPORAL_WEIGHT_GAMMA", 1.0))
        if gamma != 1.0:
            w = np.power(w, gamma)
        min_prob = float(getattr(self.cfg, "TEMPORAL_MIN_PROB", 1e-6))
        w = w + min_prob
        w = w / np.sum(w)
        self.time_weight = w
        self.time_cdf = np.cumsum(w)
        return w

    def get_tf_datasets(self):
        self._ensure_cache()

        ds = xr.open_zarr(self.cache_dir, consolidated=True)
        da_lr = ds["lr_input"]
        da_hr = ds["hr_target"]

        lr_time = next((d for d in da_lr.dims if d in ["time", "valid_time", "t"]), "time")
        lr_lat = next(
            (d for d in da_lr.dims if d in ["latitude_lr", "lat_lr", "y_lr", "latitude", "lat", "y"]),
            "y",
        )
        lr_lon = next(
            (d for d in da_lr.dims if d in ["longitude_lr", "lon_lr", "x_lr", "longitude", "lon", "x"]),
            "x",
        )
        lr_var = next((d for d in da_lr.dims if d in ["variable", "channel", "var"]), None)

        hr_time = next((d for d in da_hr.dims if d in ["time", "valid_time", "t"]), "time")
        hr_lat = next((d for d in da_hr.dims if d in ["latitude", "lat", "y"]), "y")
        hr_lon = next((d for d in da_hr.dims if d in ["longitude", "lon", "x"]), "x")

        if lr_lat not in da_lr.sizes or lr_lon not in da_lr.sizes:
            raise KeyError(f"LR dims no detectadas correctamente. dims={tuple(da_lr.dims)}")
        if hr_lat not in da_hr.sizes or hr_lon not in da_hr.sizes:
            raise KeyError(f"HR dims no detectadas correctamente. dims={tuple(da_hr.dims)}")

        hr_h = da_hr.sizes[hr_lat]
        hr_w = da_hr.sizes[hr_lon]
        lr_h = da_lr.sizes[lr_lat]
        lr_w = da_lr.sizes[lr_lon]

        if lr_var:
            lr_c = da_lr.sizes[lr_var]
        else:
            lr_c = 1

        patch = getattr(self.cfg, "PATCH_SIZE", (128, 128))
        if isinstance(patch, int):
            patch_h, patch_w = patch, patch
        else:
            patch_h, patch_w = int(patch[0]), int(patch[1])

        # Determine LR patch size by ratio
        ratio_y = lr_h / float(hr_h)
        ratio_x = lr_w / float(hr_w)
        lr_ph = max(1, int(round(patch_h * ratio_y)))
        lr_pw = max(1, int(round(patch_w * ratio_x)))

        # Update config shapes for model building
        self.cfg.HR_SHAPE = (patch_h, patch_w)
        self.cfg.LR_SHAPE = (lr_ph, lr_pw)
        self.cfg.CHANNELS = lr_c

        # Static
        static = self._load_static((hr_h, hr_w))
        self.cfg.STATIC_CHANNELS = static.shape[-1]

        # Orientation check (lon order) — uses shared mixin
        _order = TemporalSamplerMixin.order_values

        flip_lr_lon = False
        try:
            lr_lon_vals = ds[lr_lon].values
            hr_lon_vals = ds[hr_lon].values
            if _order(lr_lon_vals) != "unknown" and _order(hr_lon_vals) != "unknown":
                flip_lr_lon = _order(lr_lon_vals) != _order(hr_lon_vals)
        except Exception:
            pass

        # Splits by time
        total_len = ds.sizes[lr_time]
        seq_len = self.cfg.SEQ_LEN
        stride = int(getattr(self.cfg, "TEMPORAL_STRIDE", 1))
        if stride < 1:
            stride = 1
        if stride > seq_len:
            print(f"⚠️ TEMPORAL_STRIDE ({stride}) > SEQ_LEN ({seq_len}). Usando stride={seq_len}.")
            stride = seq_len

        # Uses shared mixin
        _time_indices = TemporalSamplerMixin.time_indices

        if getattr(self.cfg, "SPLIT_MODE", "fraction") == "time":
            times = ds[lr_time].values
            train_start, train_end = _time_indices(times, self.cfg.TRAIN_START, self.cfg.TRAIN_END)
            val_start, val_end = _time_indices(times, self.cfg.VAL_START, self.cfg.VAL_END)
        else:
            split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
            train_start, train_end = 0, split_idx
            val_start, val_end = split_idx, total_len

        sampler = getattr(self.cfg, "TILE_SAMPLER", "static_weighted")
        mix_alpha = float(getattr(self.cfg, "TILE_WEIGHT_ALPHA", 0.85))
        if sampler in ("static_weighted", "error_weighted", "uhi_proxy"):
            self._build_weight_map(sampler=sampler, hr_shape=(hr_h, hr_w))
        else:
            self.weight_cdf = None

        # Temporal weighted sampling (optional)
        self._build_time_weights(da_hr, hr_time)
        season_balance = bool(getattr(self.cfg, "TEMPORAL_SEASON_BALANCE", False))
        times_pd = pd.to_datetime(ds[lr_time].values)

        # Uses shared mixin
        _season_index = TemporalSamplerMixin.season_index

        seasons = _season_index(times_pd)

        def _sample_top_left():
            if self.weight_cdf is not None and sampler in ("static_weighted", "error_weighted", "uhi_proxy"):
                # Mix uniform and weighted sampling to avoid overfitting to a narrow spatial subset.
                if self.rng.random() < mix_alpha:
                    return self._sample_patch_top_left(hr_h, hr_w, patch_h, patch_w, sampler=sampler)
            return self._sample_patch_top_left(hr_h, hr_w, patch_h, patch_w, sampler="uniform")

        def _sample_time(start_i, end_i):
            max_start = end_i - seq_len
            if max_start <= start_i:
                return start_i

            candidates = np.arange(start_i, max_start)
            if candidates.size == 0:
                return start_i

            if season_balance:
                season_ids = [0, 1, 2, 3]
                available = [s for s in season_ids if np.any(seasons[candidates] == s)]
                if available:
                    chosen = self.rng.choice(available)
                    candidates = candidates[seasons[candidates] == chosen]
                    if candidates.size == 0:
                        candidates = np.arange(start_i, max_start)

            if self.time_cdf is not None:
                weights = self.time_weight[candidates]
                weights = weights / np.sum(weights)
                return int(self.rng.choice(candidates, p=weights))

            if sampler == "p95" and self.time_series is not None:
                values = self.time_series[candidates]
                q = float(np.nanquantile(values, 0.95))
                weights = np.maximum(values - q, 0.0)
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                min_prob = float(getattr(self.cfg, "TEMPORAL_MIN_PROB", 1e-6))
                weights = weights + min_prob
                weights = weights / np.sum(weights)
                return int(self.rng.choice(candidates, p=weights))

            return int(self.rng.choice(candidates))

        # Generator
        def generator(start_i, end_i, samples):
            for _ in range(samples):
                if end_i - start_i <= seq_len + 1:
                    break
                t0 = _sample_time(start_i, end_i)
                y0, x0 = _sample_top_left()

                # LR slice (mapped)
                lr_y0 = int(round(y0 * ratio_y))
                lr_x0 = int(round(x0 * ratio_x))
                lr_y0 = max(0, min(lr_y0, lr_h - lr_ph))
                lr_x0 = max(0, min(lr_x0, lr_w - lr_pw))
                t_idx = slice(t0, t0 + seq_len * stride, stride)

                if lr_var:
                    x_lr = da_lr.isel({lr_time: t_idx,
                                       lr_lat: slice(lr_y0, lr_y0 + lr_ph),
                                       lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
                               .transpose(lr_time, lr_lat, lr_lon, lr_var) \
                               .values
                else:
                    x_lr = da_lr.isel({lr_time: t_idx,
                                       lr_lat: slice(lr_y0, lr_y0 + lr_ph),
                                       lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
                               .transpose(lr_time, lr_lat, lr_lon) \
                               .values
                    if x_lr.ndim == 3:
                        x_lr = x_lr[..., np.newaxis]

                if flip_lr_lon:
                    x_lr = x_lr[:, :, ::-1, :]

                # HR slice
                y_hr = da_hr.isel({hr_time: t_idx,
                                   hr_lat: slice(y0, y0 + patch_h),
                                   hr_lon: slice(x0, x0 + patch_w)}) \
                           .transpose(hr_time, hr_lat, hr_lon) \
                           .values
                if y_hr.ndim == 3:
                    y_hr = y_hr[..., np.newaxis]

                # Static patch
                st_patch = static[y0:y0 + patch_h, x0:x0 + patch_w, :]
                x_st = np.broadcast_to(st_patch[np.newaxis, ...], (seq_len, *st_patch.shape))

                yield (x_lr.astype(np.float32), x_st.astype(np.float32)), y_hr.astype(np.float32)

        # Output signatures
        spec_lr = tf.TensorSpec(shape=(seq_len, lr_ph, lr_pw, lr_c), dtype=tf.float32)
        spec_st = tf.TensorSpec(shape=(seq_len, patch_h, patch_w, self.cfg.STATIC_CHANNELS), dtype=tf.float32)
        spec_hr = tf.TensorSpec(shape=(seq_len, patch_h, patch_w, 1), dtype=tf.float32)

        train_samples = int(getattr(self.cfg, "PATCHES_PER_EPOCH", 2000))
        val_samples = int(getattr(self.cfg, "VAL_PATCHES_PER_EPOCH", max(1, train_samples // 10)))

        prefetch_buf = getattr(self.cfg, "PREFETCH_BUFFER_SIZE", 2)
        if prefetch_buf in (-1, 0, "auto", "AUTOTUNE"):
            prefetch_buf = tf.data.AUTOTUNE
        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_start, train_end, train_samples),
            output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(prefetch_buf)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_start, val_end, val_samples),
            output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(prefetch_buf)

        return train_ds, val_ds
