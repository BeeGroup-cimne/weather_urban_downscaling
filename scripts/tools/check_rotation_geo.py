#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from config.runtime import Config
from src.data_loader import BigDataPipeline

#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from config.runtime import Config
from src.data_loader import BigDataPipeline
from src.models.zoo import ModelZoo
from scripts.figures.fig12_heatwave_case_study import extract_sequence

def main():
    Config.SEQ_LEN = 12
    pipe = BigDataPipeline(Config)
    ds = xr.open_zarr(pipe.cache_dir, consolidated=True)
    
    mamba = ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    weights_path = os.path.join(PROJECT_ROOT, "experiments", "models", "Ablation_MAMBA_Legacy_S42_SEQ12_best.h5")
    mamba.load_weights(weights_path)
    
    target_dt = "2017-06-28 15:00:00"
    (x_lr, x_st), y_hr_norm, lr_raw = extract_sequence(ds, target_dt, pipe, seq_len=12)
    y_pred_norm = mamba((x_lr, x_st), training=False)[0, -1, :, :, 0].numpy()
    
    # Save pure raw arrays to PNG without Matplotlib axes flipping
    plt.imsave("debug_raw_hr.png", y_hr_norm, cmap='inferno')
    plt.imsave("debug_raw_pred.png", y_pred_norm, cmap='inferno')
    print("Saved debug_raw_hr.png and debug_raw_pred.png")

if __name__ == "__main__":
    main()
