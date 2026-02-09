#!/usr/bin/env python3
"""
Entrenamiento PyTorch Mamba optimizado para GPU Server
Usa GPUServerConfig y el cache Zarr generado por el pipeline.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import xarray as xr

# Agregar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([parent_dir, os.path.join(parent_dir, 'src')])

from config.runtime import Config
from src.torch_engine.model_mamba import DownsrUNetMamba
from src.data_loader import BigDataPipeline
from src.losses import TorchHybridLoss

PROJECT_ROOT = parent_dir

class ZarrIterableDataset(IterableDataset):
    def __init__(self, cache_dir, config, split='train'):
        super().__init__()
        self.cache_dir = cache_dir
        self.cfg = config
        self.split = split
        
        # Detectar dimensiones una vez
        ds = xr.open_zarr(cache_dir, consolidated=True)
        da_lr = ds['lr_input']
        
        self.lr_lat = next((d for d in da_lr.dims if d in ['latitude_lr', 'lat_lr', 'y_lr', 'latitude', 'lat', 'y']), 'y')
        self.lr_lon = next((d for d in da_lr.dims if d in ['longitude_lr', 'lon_lr', 'x_lr', 'longitude', 'lon', 'x']), 'x')
        self.lr_time = next((d for d in da_lr.dims if d in ['time', 'valid_time', 't']), 'time')
        self.lr_var = next((d for d in da_lr.dims if d in ['variable', 'channel', 'var']), None)
        
        da_hr = ds['hr_target']
        self.hr_y = next((d for d in da_hr.dims if d in ['y', 'latitude', 'lat']), 'y')
        self.hr_x = next((d for d in da_hr.dims if d in ['x', 'longitude', 'lon']), 'x')
        
        total_len = ds.sizes[self.lr_time]

        def _time_indices(times, start, end):
            times = pd.to_datetime(times).values
            start = np.datetime64(start)
            end = np.datetime64(end)
            start_idx = int(np.searchsorted(times, start, side="left"))
            end_idx = int(np.searchsorted(times, end, side="left"))
            return start_idx, end_idx
        
        if getattr(self.cfg, "SPLIT_MODE", "fraction") == "time":
            try:
                times = ds[self.lr_time].values
                train_start, train_end = _time_indices(times, self.cfg.TRAIN_START, self.cfg.TRAIN_END)
                val_start, val_end = _time_indices(times, self.cfg.VAL_START, self.cfg.VAL_END)
            except Exception as e:
                print(f"⚠️ Time split fallback to fraction due to: {e}")
                split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
                train_start, train_end = 0, split_idx
                val_start, val_end = split_idx, total_len
        else:
            split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
            train_start, train_end = 0, split_idx
            val_start, val_end = split_idx, total_len
        
        if split == 'train':
            self.start_idx = train_start
            self.end_idx = train_end
        else:
            self.start_idx = val_start
            self.end_idx = val_end
        
        print(f"   📂 Dataset ({split}) range: {self.start_idx} -> {self.end_idx}")
        
        # Static cache
        static_path = self.cfg.STATIC_CACHE_PATH
        if not os.path.isabs(static_path):
            static_path = os.path.join(PROJECT_ROOT, static_path)
        
        if not os.path.exists(static_path):
            raise FileNotFoundError(f"No se encontró static cache: {static_path}")
        
        static_data = np.load(static_path)
        if static_data.ndim == 2:
            static_data = static_data[..., np.newaxis]
        
        mean_st = np.mean(static_data, axis=(0, 1), keepdims=True)
        std_st = np.std(static_data, axis=(0, 1), keepdims=True)
        self.static_norm = (static_data - mean_st) / (std_st + 1e-6)
        
        ds.close()
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = self.start_idx
        end = self.end_idx
        
        if worker_info is not None:
            per_worker = int(np.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = start + worker_id * per_worker
            end = min(start + per_worker, end)
        
        seq_len = self.cfg.SEQ_LEN
        
        # Abrir dataset en el worker
        ds = xr.open_zarr(self.cache_dir, consolidated=True)
        da_lr = ds['lr_input']
        da_hr = ds['hr_target']
        
        try:
            for i in range(start, end - seq_len):
                # LR Input
                if self.lr_var:
                    x_lr = da_lr.isel({self.lr_time: slice(i, i + seq_len)}) \
                                .transpose(self.lr_time, self.lr_lat, self.lr_lon, self.lr_var) \
                                .values
                else:
                    x_lr = da_lr.isel({self.lr_time: slice(i, i + seq_len)}) \
                                .transpose(self.lr_time, self.lr_lat, self.lr_lon) \
                                .values[..., np.newaxis]
                
                # (T, H, W, C) -> (T, C, H, W)
                x_lr = np.moveaxis(x_lr, -1, 1)
                
                # HR Target
                y_hr = da_hr.isel({self.lr_time: slice(i, i + seq_len)}) \
                            .transpose(self.lr_time, self.hr_y, self.hr_x) \
                            .values[..., np.newaxis]
                
                y_hr = np.moveaxis(y_hr, -1, 1)  # (T, 1, H, W)
                
                # Static (broadcast)
                x_st = np.broadcast_to(
                    self.static_norm[np.newaxis, ...],
                    (seq_len, *self.static_norm.shape)
                )
                x_st = np.moveaxis(x_st, -1, 1)  # (T, C, H, W)
                
                yield (
                    torch.from_numpy(x_lr).float(),
                    torch.from_numpy(x_st).float(),
                    torch.from_numpy(y_hr).float()
                )
        finally:
            ds.close()

def train_gpu():
    print("🚀 Iniciando entrenamiento Mamba (PyTorch GPU)...")
    
    if not hasattr(Config, "GPU_MEMORY_GB"):
        raise RuntimeError("GPU config requerido. Exporta USE_GPU_CONFIG=1 antes de ejecutar.")
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"⚙️ Dispositivo: {device}")
    
    # 1. Pipeline (cache Zarr + static)
    print("📦 Inicializando Big Data Pipeline...")
    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    
    # 2. Datasets
    num_workers = 2 if device.type == "cuda" else 0
    train_dataset = ZarrIterableDataset(Config.PATH_CACHE, Config, split='train')
    val_dataset = ZarrIterableDataset(Config.PATH_CACHE, Config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=num_workers)
    
    # 3. Model
    model = DownsrUNetMamba(
        in_channels_dyn=Config.CHANNELS,
        in_channels_static=Config.STATIC_CHANNELS,
        dim=Config.MAMBA_MODEL_DIM,
        seq_len=Config.SEQ_LEN
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = TorchHybridLoss(alpha=0.8).to(device)
    
    use_amp = Config.MIXED_PRECISION and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 4. Training loop
    accumulation_steps = max(1, Config.GRADIENT_ACCUMULATION_STEPS)
    epochs = Config.EPOCHS
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        steps = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (lr, st, target) in enumerate(train_loader):
            if Config.MAX_STEPS_PER_EPOCH is not None and steps >= Config.MAX_STEPS_PER_EPOCH:
                break
            
            lr = lr.to(device)
            st = st.to(device)
            target = target.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(lr, st)
                loss = criterion(output, target) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            steps += 1
            
            if steps % 10 == 0:
                print(f"   Epoch {epoch+1} [Step {steps}] Loss: {train_loss/steps:.6f}", end='\r')
        
        # Flush gradientes restantes
        if steps % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        print(f"\n✅ Epoch {epoch+1} | Train Loss: {train_loss/steps:.6f}")
        
        # Validación rápida
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for lr, st, target in val_loader:
                if Config.MAX_STEPS_PER_EPOCH is not None and val_steps >= Config.MAX_STEPS_PER_EPOCH:
                    break
                lr = lr.to(device)
                st = st.to(device)
                target = target.to(device)
                output = model(lr, st)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_steps += 1
        
        if val_steps > 0:
            print(f"   📊 Val Loss: {val_loss/val_steps:.6f}")
        
        # Limpieza
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    print("✅ Entrenamiento PyTorch GPU finalizado.")

if __name__ == "__main__":
    train_gpu()
