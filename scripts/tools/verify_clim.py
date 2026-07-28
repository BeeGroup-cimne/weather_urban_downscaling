import numpy as np
clim = np.load('data/processed/clim_anomaly_full.npy')
print(f'clim shape: {clim.shape}, dtype: {clim.dtype}')
static = np.load('data/processed/static_processed.npy')
print(f'static shape: {static.shape}')
if clim.shape[:2] == static.shape[:2]:
    combined = np.concatenate([static, clim[..., np.newaxis]], axis=-1)
    print(f'COMBINED OK: {combined.shape} (13 -> {combined.shape[-1]} channels)')
else:
    print(f'MISMATCH: clim {clim.shape} vs static {static.shape}')
