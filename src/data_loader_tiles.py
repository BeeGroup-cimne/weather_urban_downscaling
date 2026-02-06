# Tile-based data pipeline (independent from the full-frame pipeline).

import os
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

from config.runtime import Config
from src.data_loader import BigDataPipeline


class TileDataPipeline:
    def __init__(self, config=None):
        self.cfg = config or Config
        self.cache_dir = self.cfg.PATH_CACHE
        self.static_cache_path = self.cfg.STATIC_CACHE_PATH
        self.static_norm = None
        self.weight_map = None
        self.weight_cdf = None
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
        mean_st = np.mean(static, axis=(0, 1), keepdims=True)
        std_st = np.std(static, axis=(0, 1), keepdims=True)
        self.static_norm = (static - mean_st) / (std_st + 1e-6)
        if np.isnan(self.static_norm).any():
            self.static_norm = np.nan_to_num(self.static_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return self.static_norm

    def _build_weight_map(self, sampler="static_weighted"):
        if sampler != "static_weighted":
            return None
        static = self.static_norm
        # Importance by mean absolute normalized static signal
        weight = np.mean(np.abs(static), axis=-1)
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
        if sampler != "static_weighted" or self.weight_cdf is None:
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

    def get_tf_datasets(self):
        self._ensure_cache()

        ds = xr.open_zarr(self.cache_dir, consolidated=True)
        da_lr = ds["lr_input"]
        da_hr = ds["hr_target"]

        lr_time = next((d for d in da_lr.dims if d in ["time", "valid_time", "t"]), "time")
        lr_lat = next((d for d in da_lr.dims if d in ["latitude", "lat", "y"]), "y")
        lr_lon = next((d for d in da_lr.dims if d in ["longitude", "lon", "x"]), "x")
        lr_var = next((d for d in da_lr.dims if d in ["variable", "channel", "var"]), None)

        hr_time = next((d for d in da_hr.dims if d in ["time", "valid_time", "t"]), "time")
        hr_lat = next((d for d in da_hr.dims if d in ["latitude", "lat", "y"]), "y")
        hr_lon = next((d for d in da_hr.dims if d in ["longitude", "lon", "x"]), "x")

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

        # Orientation check (lon order)
        def _order(vals):
            diffs = np.diff(vals)
            if np.all(diffs > 0):
                return "asc"
            if np.all(diffs < 0):
                return "desc"
            return "unknown"

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

        def _time_indices(times, start, end):
            times = pd.to_datetime(times).values
            start = np.datetime64(start)
            end = np.datetime64(end)
            start_idx = int(np.searchsorted(times, start, side="left"))
            end_idx = int(np.searchsorted(times, end, side="left"))
            return start_idx, end_idx

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
        if sampler == "static_weighted":
            self._build_weight_map(sampler=sampler)
        else:
            self.weight_cdf = None

        def _sample_top_left():
            if sampler == "static_weighted" and self.weight_cdf is not None:
                # Mix uniform and weighted
                if self.rng.random() < mix_alpha:
                    return self._sample_patch_top_left(hr_h, hr_w, patch_h, patch_w, sampler="static_weighted")
            return self._sample_patch_top_left(hr_h, hr_w, patch_h, patch_w, sampler="uniform")

        # Generator
        def generator(start_i, end_i, samples):
            for _ in range(samples):
                if end_i - start_i <= seq_len + 1:
                    break
                t0 = int(self.rng.integers(start_i, end_i - seq_len))
                y0, x0 = _sample_top_left()

                # LR slice (mapped)
                lr_y0 = int(round(y0 * ratio_y))
                lr_x0 = int(round(x0 * ratio_x))
                lr_y0 = max(0, min(lr_y0, lr_h - lr_ph))
                lr_x0 = max(0, min(lr_x0, lr_w - lr_pw))

                if lr_var:
                    x_lr = da_lr.isel({lr_time: slice(t0, t0 + seq_len),
                                       lr_lat: slice(lr_y0, lr_y0 + lr_ph),
                                       lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
                               .transpose(lr_time, lr_lat, lr_lon, lr_var) \
                               .values
                else:
                    x_lr = da_lr.isel({lr_time: slice(t0, t0 + seq_len),
                                       lr_lat: slice(lr_y0, lr_y0 + lr_ph),
                                       lr_lon: slice(lr_x0, lr_x0 + lr_pw)}) \
                               .transpose(lr_time, lr_lat, lr_lon) \
                               .values
                    if x_lr.ndim == 3:
                        x_lr = x_lr[..., np.newaxis]

                if flip_lr_lon:
                    x_lr = x_lr[:, :, ::-1, :]

                # HR slice
                y_hr = da_hr.isel({hr_time: slice(t0, t0 + seq_len),
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

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_start, train_end, train_samples),
            output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(self.cfg.PREFETCH_BUFFER_SIZE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_start, val_end, val_samples),
            output_signature=((spec_lr, spec_st), spec_hr)
        ).batch(self.cfg.BATCH_SIZE, drop_remainder=True).prefetch(self.cfg.PREFETCH_BUFFER_SIZE)

        return train_ds, val_ds
