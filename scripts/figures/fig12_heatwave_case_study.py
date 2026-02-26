#!/usr/bin/env python3
"""
Generate a 3x3 Study Case visual panel for extreme Heatwave Events (2017).
Columns: Lower Resolution Input | Mamba (SEQ=12) Prediction | High Resolution Target
Rows: June 28 (Peak) | July 13 (Moderate) | August 15 (Late Summer)
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.data_loader import BigDataPipeline
from src.models_legacy import ModelZoo
import tensorflow as tf

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "presentation_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.2,
})

def _rotate(img):
    """
    Apply standard CCW 90-degree rotation (k=1) as verified in fig11.
    Additionally, apply np.fliplr to fix the East-West horizontal mirroring.
    """
    return np.fliplr(np.rot90(img, k=1))

def build_flip_lr_logic(ds):
    """Detect if LR longitude is mirrored compared to HR."""
    def order_values(arr):
        if len(arr) < 2: return "unknown"
        if arr[1] > arr[0]: return "ascending"
        elif arr[1] < arr[0]: return "descending"
        return "unknown"

    da_lr = ds['lr_input']
    da_hr = ds['hr_target']
    
    lr_lon = next((d for d in da_lr.dims if d in ['longitude_lr', 'lon_lr', 'x_lr', 'longitude', 'lon', 'x']), 'x')
    hr_x = next((d for d in da_hr.dims if d in ['x', 'longitude', 'lon']), 'x')

    try:
        lr_lon_vals = ds[lr_lon].values
        hr_lon_vals = ds[hr_x].values
        lr_order = order_values(lr_lon_vals)
        hr_order = order_values(hr_lon_vals)
        if lr_order != "unknown" and hr_order != "unknown" and lr_order != hr_order:
            return True, lr_lon
        return False, lr_lon
    except Exception:
        return False, lr_lon

def extract_sequence(ds, target_dt, pipe, seq_len=12):
    # Match the exact hour
    times = pd.to_datetime(ds.time.values).floor('h')
    matches = np.where(times == pd.to_datetime(target_dt))[0]
    
    if len(matches) == 0:
        raise ValueError(f"No snapshot found for {target_dt}")
    
    end_idx = matches[0]
    start_idx = end_idx - (seq_len - 1)
    
    if start_idx < 0:
        raise ValueError(f"Not enough history (seq_len={seq_len}) before {target_dt}")
    
    t_slice = slice(start_idx, end_idx + 1)
    
    # ── LR Data ──
    da_lr = ds['lr_input']
    lr_dims = list(da_lr.dims)
    lr_lat = next((d for d in lr_dims if d in ['latitude_lr', 'lat_lr', 'y_lr', 'latitude', 'lat', 'y']), 'y')
    
    flip_lr_lon, lr_lon = build_flip_lr_logic(ds)
    
    if 'variable' in da_lr.dims:
        x_lr = da_lr.isel(time=t_slice).transpose('time', lr_lat, lr_lon, 'variable').values
    else:
        x_lr = da_lr.isel(time=t_slice).transpose('time', lr_lat, lr_lon).values
        if x_lr.ndim == 3: x_lr = x_lr[..., np.newaxis]
        
    if flip_lr_lon: x_lr = x_lr[:, :, ::-1, :]
    
    # Expand batch
    x_lr = x_lr[np.newaxis, ...]
    
    # ── HR Data (Target) ──
    da_hr = ds['hr_target']
    hr_y = next((d for d in da_hr.dims if d in ['y', 'latitude', 'lat']), 'y')
    hr_x = next((d for d in da_hr.dims if d in ['x', 'longitude', 'lon']), 'x')
    
    # Target is only the EXACT moment (end_idx)
    y_hr = da_hr.isel(time=end_idx).transpose(hr_y, hr_x).values
    
    # ── STATIC Data ──
    static_data = pipe.ds_static_single
    if static_data.ndim == 2: static_data = static_data[..., np.newaxis]
    mean_st = np.mean(static_data, axis=(0, 1), keepdims=True)
    std_st = np.std(static_data, axis=(0, 1), keepdims=True)
    static_norm = (static_data - mean_st) / (std_st + 1e-6)
    
    x_st = np.broadcast_to(static_norm[np.newaxis, np.newaxis, ...], 
                           (1, seq_len) + static_norm.shape)
                           
    return (x_lr, x_st), y_hr, x_lr[0, -1, ...] # Retornamos LR de la capa temporal final

def main():
    print("🚀 Initializing Heatwave Case Study Predictor (MAMBA SEQ=12)...")
    
    # Pipeline Init to generate cache and stats if needed
    Config.SEQ_LEN = 12
    pipe = BigDataPipeline(Config)
    pipe.process_static_data()
    pipe.run_etl_process()
    
    # IMPORTANTE: Llamar a get_tf_datasets detecta dinámicamente LR_SHAPE (ej: 5x4)
    # y lo setea en Config.LR_SHAPE. Si no hacemos esto, inicializamos Mamba con 251x251
    # y los pesos cargados que traen la capa UpSampling provocarán incompatibilidad,
    # o bien tendremos que interpolar nosotros a nearest rompiendo el feature space.
    pipe.get_tf_datasets()
    
    # Load model (Con Config.LR_SHAPE correctamente actualizado)
    mamba = ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    weights_path = os.path.join(PROJECT_ROOT, "experiments", "models", "Ablation_MAMBA_Legacy_S42_SEQ12_best.h5")
    mamba.load_weights(weights_path)
    
    # Zarr Dataset and Stats
    ds = xr.open_zarr(pipe.cache_dir, consolidated=True)
    stats = np.load(Config.STATS_PATH)
    mean_hr, std_hr = float(stats["mean_hr"]), float(stats["std_hr"])
    
    # Identificar canal de Temperatura en LR
    variables = ds['lr_input'].variable.values.tolist() if 'variable' in ds['lr_input'].dims else []
    t2m_idx = variables.index("t2m") if "t2m" in variables else 0
    mean_lr = float(stats["mean_lr"][t2m_idx] if np.ndim(stats["mean_lr"]) > 0 else stats["mean_lr"])
    std_lr = float(stats["std_lr"][t2m_idx] if np.ndim(stats["std_lr"]) > 0 else stats["std_lr"])

    # 3 heatwave episodes: Jun 28 (peak), Jul 13 (moderate), Aug 15 (late summer)
    episodes = [
        ("2017-06-28 15:00:00", "28 Jun 15:00\n(Peak Heatwave)"),
        ("2017-07-13 16:00:00", "13 Jul 16:00\n(Moderate Heatwave)"),
        ("2017-08-15 15:00:00", "15 Aug 15:00\n(Late Summer Heat)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Urban Heatwave Downscaling: Spatial Integrity of Mamba (SEQ=12) v3", fontsize=22, fontweight='bold', y=0.94)

    cmap = "inferno"

    for row, (target_dt, label) in enumerate(episodes):
        print(f"🌡️ Processing Episode {row + 1}/3: {label.replace(chr(10), ' ')}...")
        
        # Inferencia
        (x_lr, x_st), y_hr_norm, lr_raw_norm = extract_sequence(ds, target_dt, pipe, seq_len=12)
        y_pred_norm = mamba((x_lr, x_st), training=False)[0, -1, :, :, 0].numpy()
        
        # Des-normalizar HR y PREDICTION
        y_hr = y_hr_norm * std_hr + mean_hr
        y_pred = y_pred_norm * std_hr + mean_hr
        
        # Des-normalizar LR usando t2m_idx
        # lr_raw_norm shape es (5, 4, Canales) tal cual sale de ERA5Land
        print(f"DEBUG LR RAW -> shape: {lr_raw_norm.shape}, min: {lr_raw_norm.min():.2f}, max: {lr_raw_norm.max():.2f}")
        lr_t2m = lr_raw_norm[..., t2m_idx] * std_lr + mean_lr
        
        # ── INTERPOLACIÓN NEAREST para Visualización ESTÉTICA ──
        # Aquí reconstruimos los píxeles cuadradotes originales (5x4 -> 251x251) 
        # exclusivamente para hacer el cuadrito del gráfico sin que matplotlib lo difumine.
        lr_up = tf.image.resize(lr_t2m[..., np.newaxis], Config.HR_SHAPE, method="nearest").numpy()[..., 0]
        
        # Rango global robusto para HR y Predicción
        vmin = min(y_hr.min(), y_pred.min())
        vmax = max(y_hr.max(), y_pred.max())
        
        # Rango para LR independiente (por si hay un bug de escala que lo está volviendo negro)
        lr_vmin = lr_up.min()
        lr_vmax = lr_up.max()

        # Subplots 
        # (1) LR Original Interpolado
        ax_lr = axes[row, 0]
        ax_lr.imshow(_rotate(lr_up), cmap=cmap, origin='lower', vmin=lr_vmin, vmax=lr_vmax, interpolation='nearest')
        if row == 0: ax_lr.set_title(f"Low Resolution Input\nMin: {lr_vmin:.1f}, Max: {lr_vmax:.1f}", fontsize=16)
        ax_lr.set_ylabel(label, fontsize=16, fontweight='bold', labelpad=20)
        ax_lr.set_xticks([])
        ax_lr.set_yticks([])

        # (2) Predicción Mamba
        mae = np.mean(np.abs(y_pred - y_hr))
        ax_pred = axes[row, 1]
        im = ax_pred.imshow(_rotate(y_pred), cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        if row == 0: ax_pred.set_title("MAMBA Prediction (SEQ=12)\nHigh Res Downscaling", fontsize=16)
        metrics_text = f"MAE: {mae:.2f} °C"
        ax_pred.text(0.05, 0.05, metrics_text, transform=ax_pred.transAxes, color="white", fontsize=14, 
                     fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
        ax_pred.axis("off")

        # (3) Target High Resolution
        ax_target = axes[row, 2]
        ax_target.imshow(_rotate(y_hr), cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        if row == 0: ax_target.set_title("Ground Truth\nUrban Weather Stations", fontsize=16)
        ax_target.axis("off")

    # Colorbar común en la parte inferior o la derecha
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Temperature T2m (°C)")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    out_path = os.path.join(OUT_DIR, "fig12_heatwave_case_study.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Generated {out_path}")

if __name__ == "__main__":
    main()
