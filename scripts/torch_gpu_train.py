#!/usr/bin/env python3
"""
Entrenamiento PyTorch Mamba optimizado para GPU Server
Usa GPUServerConfig y el cache Zarr generado por el pipeline.
"""

import os
import sys
import gc
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _match_pred_to_target_spatial(pred, target):
    """
    Ensure prediction matches target spatial resolution.
    Expected tensors: (B, T, C, H, W)
    """
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    b, t, c, h, w = pred.shape
    target_h, target_w = target.shape[-2:]
    pred_4d = pred.view(b * t, c, h, w)
    pred_4d = F.interpolate(pred_4d, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return pred_4d.view(b, t, c, target_h, target_w)


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_amp_scaler(use_amp):
    if not use_amp:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(use_amp):
    if not use_amp:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _safe_torch_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_torch_checkpoint(path, model, optimizer, scaler, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return state

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
                x_st = np.ascontiguousarray(x_st)
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
    scaler = _build_amp_scaler(use_amp)
    if use_amp:
        print("⚡ AMP activado para CUDA")
    else:
        print("ℹ️ AMP desactivado")

    experiment_name = getattr(Config, "MAMBA_EXPERIMENT_NAME", f"mamba_seq{Config.SEQ_LEN}")
    model_dir = os.path.join(Config.EXPERIMENTS_DIR, "models")
    logs_dir = os.path.join(Config.EXPERIMENTS_DIR, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ckpt_last = os.path.join(model_dir, f"{experiment_name}_last.pt")
    ckpt_best = os.path.join(model_dir, f"{experiment_name}_best.pt")
    history_csv = os.path.join(logs_dir, f"{experiment_name}_torch_log.csv")

    start_epoch = 0
    best_val = float("inf")
    history_rows = []

    if os.path.exists(history_csv):
        try:
            df_hist = pd.read_csv(history_csv)
            history_rows = df_hist.to_dict(orient="records")
        except Exception as e:
            print(f"⚠️ No se pudo cargar histórico CSV: {e}")

    if _truthy(os.getenv("RESUME_TRAINING", "1")):
        resume_path = os.getenv("RESUME_CHECKPOINT", "").strip() or ckpt_last
        if os.path.exists(resume_path):
            try:
                state = _load_torch_checkpoint(resume_path, model, optimizer, scaler, device)
                start_epoch = int(state.get("epoch", 0))
                best_val = float(state.get("best_val_loss", best_val))
                print(f"🔁 Resume activo desde: {resume_path}")
                print(f"   Continuando en epoch {start_epoch + 1}")
            except Exception as e:
                print(f"⚠️ No se pudo cargar checkpoint ({resume_path}): {e}")
        else:
            print(f"ℹ️ Resume habilitado pero no existe checkpoint: {resume_path}")
    
    # 4. Training loop
    accumulation_steps = max(1, Config.GRADIENT_ACCUMULATION_STEPS)
    epochs = Config.EPOCHS
    
    if start_epoch >= epochs:
        print(f"✅ Entrenamiento ya completado ({start_epoch}/{epochs} epochs).")
        return

    for epoch in range(start_epoch, epochs):
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
            
            with _autocast_context(use_amp):
                output = model(lr, st)
                output = _match_pred_to_target_spatial(output, target)
                loss = criterion(output, target) / accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            steps += 1
            
            if steps % 10 == 0:
                print(f"   Epoch {epoch+1} [Step {steps}] Loss: {train_loss/steps:.6f}", end='\r')
        
        # Flush gradientes restantes
        if steps % accumulation_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        avg_train = train_loss / max(1, steps)
        print(f"\n✅ Epoch {epoch+1} | Train Loss: {avg_train:.6f}")
        
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
                output = _match_pred_to_target_spatial(output, target)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_steps += 1
        
        avg_val = float("nan")
        if val_steps > 0:
            avg_val = val_loss / val_steps
            print(f"   📊 Val Loss: {avg_val:.6f}")
        
        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val,
            "experiment_name": experiment_name,
            "seq_len": Config.SEQ_LEN,
            "batch_size": Config.BATCH_SIZE,
            "grad_accumulation": accumulation_steps,
            "learning_rate": Config.LEARNING_RATE,
        }
        _safe_torch_save(checkpoint_payload, ckpt_last)

        if val_steps > 0 and avg_val < best_val:
            best_val = avg_val
            checkpoint_payload["best_val_loss"] = best_val
            _safe_torch_save(checkpoint_payload, ckpt_best)
            print(f"   💾 Nuevo best checkpoint: {ckpt_best} (val={best_val:.6f})")

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "seq_len": Config.SEQ_LEN,
                "batch_size": Config.BATCH_SIZE,
                "grad_accumulation": accumulation_steps,
            }
        )
        try:
            pd.DataFrame(history_rows).to_csv(history_csv, index=False)
        except Exception as e:
            print(f"⚠️ No se pudo guardar histórico CSV: {e}")
        
        print(f"   💾 Last checkpoint: {ckpt_last}")
        
        # Limpieza
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    print("✅ Entrenamiento PyTorch GPU finalizado.")

if __name__ == "__main__":
    train_gpu()
