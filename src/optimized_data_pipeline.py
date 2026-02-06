"""
BigDataPipeline optimizado para GPU server con memoria limitada
Solución a problemas de OOM mediante chunking y broadcasting eficiente
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from dask.diagnostics import ProgressBar
from typing import Tuple, Optional
import gc
try:
    import psutil
except ImportError:
    psutil = None

# Importar configuración
from config.runtime import Config

class OptimizedBigDataPipeline:
    def __init__(self, config=None):
        self.config = config or Config
        self.stats = None
        self.static_processed = None
        
        # Memory monitoring
        self.memory_peak = 0
        
        print(f"🚀 Pipeline Optimizado inicializado")
        print(f"   Batch Size: {self.config.BATCH_SIZE}")
        print(f"   Chunk Size: {self.config.ZARR_CHUNK_SIZE}")
        
        # Cargar cache estático si existe
        if os.path.exists(self.config.STATIC_CACHE_PATH):
            try:
                self.static_processed = np.load(self.config.STATIC_CACHE_PATH)
                print(f"✅ Static cache cargado: {self.config.STATIC_CACHE_PATH}")
            except Exception as e:
                print(f"⚠️ Error cargando static cache: {e}")
                self.static_processed = None
        
    def monitor_memory(self, stage: str):
        """Monitor memory usage con logging"""
        gpu_memory = 0
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1024**3
        except:
            pass
            
        cpu_memory = psutil.virtual_memory().percent if psutil else None
        self.memory_peak = max(self.memory_peak, gpu_memory)
        
        if cpu_memory is None:
            print(f"📊 [{stage}] GPU: {gpu_memory:.2f}GB (Peak GPU: {self.memory_peak:.2f}GB)")
        else:
            print(f"📊 [{stage}] GPU: {gpu_memory:.2f}GB, CPU: {cpu_memory}% (Peak GPU: {self.memory_peak:.2f}GB)")
        
    def process_static_data(self):
        """Procesamiento de datos estáticos con lógica robusta (reusa pipeline base)."""
        print("🏗️ Procesando datos estáticos (optimizado)...")
        
        # 1) Si ya existe cache, úsalo
        if os.path.exists(self.config.STATIC_CACHE_PATH):
            try:
                self.static_processed = np.load(self.config.STATIC_CACHE_PATH)
                print(f"✅ Static cache cargado: {self.config.STATIC_CACHE_PATH}")
                return
            except Exception as e:
                print(f"⚠️ Error cargando static cache: {e}")
        
        # 2) Fallback: generar cache usando el pipeline base (interpolación robusta)
        try:
            print("🧩 Generando static cache con BigDataPipeline base...")
            from src.data_loader import BigDataPipeline as BasePipeline
            base_pipeline = BasePipeline(self.config)
            base_pipeline.process_static_data()
            
            if os.path.exists(self.config.STATIC_CACHE_PATH):
                self.static_processed = np.load(self.config.STATIC_CACHE_PATH)
                print(f"✅ Static cache generado: {self.config.STATIC_CACHE_PATH}")
            else:
                # Último fallback: usar los datos procesados en memoria
                self.static_processed = base_pipeline.ds_static_single
                if self.static_processed is not None:
                    np.save(self.config.STATIC_CACHE_PATH, self.static_processed)
                    print(f"✅ Static cache guardado: {self.config.STATIC_CACHE_PATH}")
            
        except Exception as e:
            print(f"❌ Error procesando datos estáticos: {e}")
            raise
            
    def _normalize_static_data(self, ds_static):
        """Normalización eficiente de datos estáticos"""
        # Convertir a numpy una sola vez
        static_array = ds_static.to_array().values  # Shape: (vars, lat, lon)
        
        # Transponer a (lat, lon, vars)
        static_array = np.transpose(static_array, (1, 2, 0))
        
        # Normalizar por canal
        static_norm = np.zeros_like(static_array, dtype=np.float32)
        
        for i in range(static_array.shape[2]):
            channel_data = static_array[:, :, i]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            if std_val > 0:
                static_norm[:, :, i] = (channel_data - mean_val) / std_val
            else:
                static_norm[:, :, i] = channel_data - mean_val
                
        return static_norm.astype(np.float32)
    
    def get_tf_datasets(self, include_test: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Crear TensorFlow datasets con optimización de memoria (streaming)"""
        print("📦 Creando TF datasets optimizados...")
        
        try:
            # Cargar datos dinámicos con chunking
            ds = xr.open_zarr(
                self.config.PATH_CACHE,
                chunks={'time': self.config.ZARR_CHUNK_SIZE}
            )
            
            self.monitor_memory("Dynamic Load")
            
            # Analizar dims de LR/HR
            da_lr = ds['lr_input']
            da_hr = ds['hr_target']
            
            lr_dims = list(da_lr.dims)
            lr_time = next((d for d in lr_dims if d in ['time', 'valid_time', 't']), 'time')
            lr_lat = next((d for d in lr_dims if d in ['latitude', 'lat', 'y']), 'y')
            lr_lon = next((d for d in lr_dims if d in ['longitude', 'lon', 'x']), 'x')
            lr_var = next((d for d in lr_dims if d in ['variable', 'channel', 'var']), None)
            
            # Ajustar configuración LR/CHANNELS
            real_lr_h = da_lr.sizes[lr_lat]
            real_lr_w = da_lr.sizes[lr_lon]
            if lr_var:
                real_lr_c = da_lr.sizes[lr_var]
            else:
                real_lr_c = 1
            
            if self.config.LR_SHAPE != (real_lr_h, real_lr_w):
                print(f"⚠️ LR_SHAPE: {self.config.LR_SHAPE} -> ({real_lr_h}, {real_lr_w})")
                self.config.LR_SHAPE = (real_lr_h, real_lr_w)
            if self.config.CHANNELS != real_lr_c:
                print(f"⚠️ CHANNELS: {self.config.CHANNELS} -> {real_lr_c}")
                self.config.CHANNELS = real_lr_c
            
            hr_dims = list(da_hr.dims)
            hr_time = next((d for d in hr_dims if d in ['time', 'valid_time', 't']), 'time')
            hr_y = next((d for d in hr_dims if d in ['y', 'latitude', 'lat']), 'y')
            hr_x = next((d for d in hr_dims if d in ['x', 'longitude', 'lon']), 'x')
            
            real_hr_h = da_hr.sizes[hr_y]
            real_hr_w = da_hr.sizes[hr_x]
            if self.config.HR_SHAPE != (real_hr_h, real_hr_w):
                print(f"⚠️ HR_SHAPE: {self.config.HR_SHAPE} -> ({real_hr_h}, {real_hr_w})")
                self.config.HR_SHAPE = (real_hr_h, real_hr_w)

            # Detect longitude order mismatch and align LR with HR
            def _order(vals):
                diffs = np.diff(vals)
                if np.all(diffs > 0):
                    return "asc"
                if np.all(diffs < 0):
                    return "desc"
                return "unknown"

            try:
                lr_lon_vals = ds[lr_lon].values
                hr_lon_vals = ds[hr_x].values
                lr_order = _order(lr_lon_vals)
                hr_order = _order(hr_lon_vals)
            except Exception:
                lr_order = hr_order = "unknown"

            flip_lr_lon = False
            if lr_order != "unknown" and hr_order != "unknown" and lr_order != hr_order:
                flip_lr_lon = True
                print(f"⚠️ Longitud order mismatch (LR={lr_order}, HR={hr_order}). Aplicando flip LR en lon.")
            
            # Cargar datos estáticos
            if self.static_processed is None:
                try:
                    self.static_processed = np.load(self.config.STATIC_CACHE_PATH)
                except:
                    print("⚠️ Cache estático no encontrado, procesando desde cero...")
                    self.process_static_data()
            
            if self.static_processed is None:
                raise RuntimeError("❌ No se pudo cargar datos estáticos.")
            
            # Normalizar estáticos
            static_data = self.static_processed
            if static_data.ndim == 2:
                static_data = static_data[..., np.newaxis]
            
            mean_st = np.mean(static_data, axis=(0, 1), keepdims=True)
            std_st = np.std(static_data, axis=(0, 1), keepdims=True)
            static_norm = (static_data - mean_st) / (std_st + 1e-6)
            
            # Ajustar canales estáticos si difieren
            if static_norm.shape[-1] != self.config.STATIC_CHANNELS:
                print(f"⚠️ STATIC_CHANNELS: {self.config.STATIC_CHANNELS} -> {static_norm.shape[-1]}")
                self.config.STATIC_CHANNELS = static_norm.shape[-1]
            
            if static_norm.shape[0] != self.config.HR_SHAPE[0] or static_norm.shape[1] != self.config.HR_SHAPE[1]:
                print(f"⚠️ Static shape {static_norm.shape[:2]} no coincide con HR_SHAPE {self.config.HR_SHAPE}")
            
            total_len = ds.sizes[lr_time]
            seq_len = self.config.SEQ_LEN
            stride = int(getattr(self.config, "TEMPORAL_STRIDE", 1))
            if stride < 1:
                stride = 1
            if stride > seq_len:
                print(f"⚠️ TEMPORAL_STRIDE ({stride}) > SEQ_LEN ({seq_len}). Usando stride={seq_len}.")
                stride = seq_len
            temporal_sampler = getattr(self.config, "TEMPORAL_SAMPLER", "uniform")
            season_balance = bool(getattr(self.config, "TEMPORAL_SEASON_BALANCE", False))
            rng = np.random.default_rng(getattr(self.config, "SEED", 42))

            def _time_indices(times, start, end):
                times = pd.to_datetime(times).values
                start = np.datetime64(start)
                end = np.datetime64(end)
                start_idx = int(np.searchsorted(times, start, side="left"))
                end_idx = int(np.searchsorted(times, end, side="left"))
                return start_idx, end_idx

            test_start = test_end = None
            if getattr(self.config, "SPLIT_MODE", "fraction") == "time":
                try:
                    times = ds[lr_time].values
                    train_start, train_end = _time_indices(times, self.config.TRAIN_START, self.config.TRAIN_END)
                    val_start, val_end = _time_indices(times, self.config.VAL_START, self.config.VAL_END)
                    if include_test:
                        test_start, test_end = _time_indices(times, self.config.TEST_START, self.config.TEST_END)
                    print(f"   📂 Dataset (train) range: {train_start} -> {train_end}")
                    print(f"   📂 Dataset (val) range: {val_start} -> {val_end}")
                    if include_test:
                        print(f"   📂 Dataset (test) range: {test_start} -> {test_end}")
                except Exception as e:
                    print(f"⚠️ Time split fallback to fraction due to: {e}")
                    split_idx = int(total_len * self.config.SPLIT_FRACTION)
                    train_start, train_end = 0, split_idx
                    val_start, val_end = split_idx, total_len
            else:
                split_idx = int(total_len * self.config.SPLIT_FRACTION)
                train_start, train_end = 0, split_idx
                if include_test:
                    val_end = int(total_len * 0.9)
                    val_start = split_idx
                    test_start, test_end = val_end, total_len
                else:
                    val_start, val_end = split_idx, total_len
            
            # Precompute valid start indices
            max_start_train = train_end - seq_len * stride
            max_start_val = val_end - seq_len * stride
            max_start_test = test_end - seq_len * stride if include_test else None

            # Temporal weights (optional)
            def _build_time_weights():
                if temporal_sampler not in ("weighted", "weighted_station"):
                    return None
                series = None
                if temporal_sampler == "weighted_station":
                    station_path = getattr(self.config, "STATION_GRIB_PATH", "")
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
                            time_dim = next((d for d in da.dims if d in ["time", "valid_time"]), da.dims[0])
                            reduce_dims = [d for d in da.dims if d != time_dim]
                            series = da.mean(dim=reduce_dims).values
                            st_times = pd.to_datetime(da[time_dim].values).floor("H")
                            ds_times = pd.to_datetime(ds[lr_time].values).floor("H")
                            time_map = {t: i for i, t in enumerate(st_times)}
                            aligned = np.zeros(ds_times.shape[0], dtype=np.float32)
                            for i, t in enumerate(ds_times):
                                j = time_map.get(t)
                                if j is not None:
                                    aligned[i] = series[j]
                            series = aligned
                        except Exception:
                            series = None
                    if series is None:
                        print("⚠️ No se pudo usar estaciones para pesos temporales, fallback a HR.")

                if series is None:
                    try:
                        hr_mean = da_hr.mean(dim=[d for d in da_hr.dims if d != hr_time]).values
                        series = hr_mean
                    except Exception:
                        series = None
                if series is None:
                    return None

                series = np.asarray(series, dtype=np.float32)
                grad = np.abs(np.diff(series, prepend=series[0]))
                grad = grad - np.nanmin(grad)
                gamma = float(getattr(self.config, "TEMPORAL_WEIGHT_GAMMA", 1.0))
                if gamma != 1.0:
                    grad = np.power(grad, gamma)
                min_prob = float(getattr(self.config, "TEMPORAL_MIN_PROB", 1e-6))
                grad = grad + min_prob
                grad = grad / np.sum(grad)
                return grad

            time_weights = _build_time_weights()
            times_pd = pd.to_datetime(ds[lr_time].values)

            def _season_index(times_idx):
                months = times_idx.month
                seasons = np.zeros_like(months)
                seasons[(months >= 3) & (months <= 5)] = 1
                seasons[(months >= 6) & (months <= 8)] = 2
                seasons[(months >= 9) & (months <= 11)] = 3
                return seasons

            seasons = _season_index(times_pd)

            def _sample_time(start_i, end_i, max_start):
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
                        chosen = rng.choice(available)
                        candidates = candidates[seasons[candidates] == chosen]
                if time_weights is None:
                    return int(rng.choice(candidates))
                weights = time_weights[candidates]
                weights = weights / np.sum(weights)
                return int(rng.choice(candidates, p=weights))

            # Generador streaming para evitar OOM
            def generator(start_i, end_i, max_start):
                if temporal_sampler != "uniform" or season_balance:
                    sample_count = max(1, max_start - start_i)
                    for _ in range(sample_count):
                        i = _sample_time(start_i, end_i, max_start)
                        t_idx = slice(i, i + seq_len * stride, stride)
                        yield _yield_at(t_idx)
                    return
                for i in range(start_i, end_i - seq_len * stride):
                    t_idx = slice(i, i + seq_len * stride, stride)
                    yield _yield_at(t_idx)

            def _yield_at(t_idx):
                if lr_var:
                    x_lr = da_lr.isel({lr_time: t_idx}) \
                        .transpose(lr_time, lr_lat, lr_lon, lr_var) \
                        .values
                else:
                    x_lr = da_lr.isel({lr_time: t_idx}) \
                        .transpose(lr_time, lr_lat, lr_lon) \
                        .values
                    if x_lr.ndim == 3:
                        x_lr = x_lr[..., np.newaxis]

                if flip_lr_lon:
                    x_lr = x_lr[:, :, ::-1, :]
                
                y_hr = da_hr.isel({hr_time: t_idx}) \
                    .transpose(hr_time, hr_y, hr_x) \
                    .values
                if y_hr.ndim == 3:
                    y_hr = y_hr[..., np.newaxis]
                
                x_st = np.broadcast_to(
                    static_norm[np.newaxis, ...],
                    (seq_len, *static_norm.shape)
                )
                
                return (x_lr, x_st), y_hr
            
            # Output signatures
            lr_h, lr_w = self.config.LR_SHAPE
            st_h, st_w = self.config.HR_SHAPE
            n_channels = self.config.CHANNELS
            
            spec_lr = tf.TensorSpec(shape=(seq_len, lr_h, lr_w, n_channels), dtype=tf.float32)
            spec_st = tf.TensorSpec(shape=(seq_len, st_h, st_w, self.config.STATIC_CHANNELS), dtype=tf.float32)
            spec_hr = tf.TensorSpec(shape=(seq_len, st_h, st_w, 1), dtype=tf.float32)
            
            shuffle_buf = getattr(self.config, "SHUFFLE_BUFFER_SIZE", 100)
            prefetch_buf = getattr(self.config, "PREFETCH_BUFFER_SIZE", 2)
            
            train_ds = tf.data.Dataset.from_generator(
                lambda: generator(train_start, train_end, max_start_train),
                output_signature=((spec_lr, spec_st), spec_hr)
            ).shuffle(shuffle_buf).batch(self.config.BATCH_SIZE, drop_remainder=True) \
             .prefetch(prefetch_buf)
            
            val_ds = tf.data.Dataset.from_generator(
                lambda: generator(val_start, val_end, max_start_val),
                output_signature=((spec_lr, spec_st), spec_hr)
            ).batch(self.config.BATCH_SIZE, drop_remainder=True).prefetch(prefetch_buf)

            if include_test:
                test_ds = tf.data.Dataset.from_generator(
                    lambda: generator(test_start, test_end, max_start_test),
                    output_signature=((spec_lr, spec_st), spec_hr)
                ).batch(self.config.BATCH_SIZE, drop_remainder=True).prefetch(prefetch_buf)
                print(f"✅ Datasets creados: Train/Val/Test streaming")
                return train_ds, val_ds, test_ds

            print(f"✅ Datasets creados: Train/Val streaming")
            return train_ds, val_ds
            
        except Exception as e:
            print(f"❌ Error creando datasets: {e}")
            raise
    
    def _prepare_sequences_efficient(self, lr_data, hr_data, static_norm):
        """Preparar secuencias con broadcasting eficiente"""
        print("🔄 Preparando secuencias (broadcasting eficiente)...")
        
        seq_len = self.config.SEQ_LEN
        sequences_lr = []
        sequences_st = []
        sequences_y = []
        
        # Limitar cantidad de secuencias para evitar OOM en debugging
        max_sequences_cfg = getattr(self.config, "MAX_SEQUENCES", None)
        max_sequences = len(lr_data) - seq_len + 1
        if max_sequences_cfg is not None:
            max_sequences = min(max_sequences_cfg, max_sequences)
        
        for i in range(max_sequences):
            try:
                # Features dinámicos
                x_dynamic = lr_data[i:i+seq_len]  # (seq_len, lr_shape, channels)
                
                # Broadcasting eficiente de datos estáticos
                # Evitar np.repeat que duplica memoria
                x_static = np.broadcast_to(
                    static_norm[np.newaxis, ...],
                    (seq_len, *static_norm.shape)
                )
                
                # Target
                y_sequence = hr_data[i:i+seq_len]  # Secuencia completa como target
                
                sequences_lr.append(x_dynamic.astype(np.float32))
                sequences_st.append(x_static.astype(np.float32))
                sequences_y.append(y_sequence.astype(np.float32))
                
                # Limpiar variables temporales
                del x_dynamic, x_static, y_sequence
                
                if i % 100 == 0:
                    print(f"   Procesadas {i}/{max_sequences} secuencias...")
                    self.monitor_memory(f"Seq {i}")
                    
            except Exception as e:
                print(f"⚠️ Error en secuencia {i}: {e}")
                continue
        
        return np.array(sequences_lr), np.array(sequences_st), np.array(sequences_y)
    
    def _create_optimized_dataset(self, x_lr, x_st, y_data, training=True):
        """Crear TF dataset con configuración de memoria optimizada"""
        dataset = tf.data.Dataset.from_tensor_slices(((x_lr, x_st), y_data))
        
        if training:
            # Shuffle con buffer pequeño para ahorrar memoria
            dataset = dataset.shuffle(
                buffer_size=min(self.config.SHUFFLE_BUFFER_SIZE, len(x_lr)),
                seed=self.config.SEED
            )
        
        # Batch con prefetch pequeño
        dataset = dataset.batch(self.config.BATCH_SIZE, drop_remainder=True)
        
        # Prefetch limitado para evitar acumulación
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE if self.config.GPU_MEMORY_GB and self.config.GPU_MEMORY_GB >= 24 else 1
        )
        
        # Mixed precision si está disponible
        if self.config.MIXED_PRECISION:
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("🎯 Mixed precision activado")
            except:
                print("⚠️ Mixed precision no disponible")
        
        return dataset

# Función wrapper para compatibilidad con código existente
def BigDataPipeline(config=None):
    """Wrapper para compatibilidad"""
    return OptimizedBigDataPipeline(config)

if __name__ == "__main__":
    print("🧪 Testing Optimized Pipeline...")
    pipeline = OptimizedBigDataPipeline()
    
    # Test
    pipeline.process_static_data()
    train_ds, val_ds = pipeline.get_tf_datasets()
    
    print("✅ Pipeline optimizado funcionando correctamente!")
