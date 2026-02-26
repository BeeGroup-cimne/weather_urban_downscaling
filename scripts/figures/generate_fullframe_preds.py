#!/usr/bin/env python3
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from src.data_loader import BigDataPipeline
from src.models_legacy import ModelZoo

def load_stats():
    stats = np.load(Config.STATS_PATH)
    return float(stats["mean_hr"]), float(stats["std_hr"])

def build_model(model_type: str):
    if model_type == "mamba":
        return ModelZoo.build_hybrid_unet_mamba(lr_shape=Config.LR_SHAPE, hr_shape=Config.HR_SHAPE)
    elif model_type == "convlstm":
        return ModelZoo.build_hybrid_unet_lstm()
    elif model_type == "transformer":
        return ModelZoo.build_transformer()
    else:
        return ModelZoo.build_unet()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    # Time of interest
    target_time = "2017-06-28_15_00_00"
    
    # We want to use BigDataPipeline
    Config.SPLIT_MODE = "time"
    Config.TEST_START = "2017-06-28"
    Config.TEST_END = "2017-06-29"

    mean_hr, std_hr = load_stats()
    
    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    _, _, test_ds = pipeline.get_tf_datasets(include_test=True)
    
    # Find the specific batch/index corresponding to target_time?
    # Our data pipeline yields batches of sequences.
    # 2017-06-28 is the first day of the split.
    # 15:00 is index 15 in the first day.
    
    # Let's just run through the dataset and extract what we need.
    target_time_idx = 15  # For 15:00
    
    # Since seq_len is 6 (based on Config.SEQ_LEN), index 15 will be in 
    # batch = 15 // 6 = 2, seq_idx = 15 % 6 = 3 (assuming stride=6)
    # Stride is 6, so 1st batch: 0-5. 2nd batch: 6-11. 3rd batch: 12-17.
    # Therefore target index is in Batch 2, seq index 3.
    target_batch = target_time_idx // Config.SEQ_LEN
    target_seq_idx = target_time_idx % Config.SEQ_LEN
    
    # Advance dataset to target_batch
    ds_iter = iter(test_ds)
    for _ in range(target_batch):
        next(ds_iter)
        
    (x_lr, x_st), y_true = next(ds_iter)
    
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "features_fullframe")
    os.makedirs(out_dir, exist_ok=True)
    
    if args.baseline:
        import tensorflow as tf
        
        # x_lr is 5D: (batch, seq_len, h, w, c). Extract the seq_len to get 4D: (seq_len, h, w, c)
        x_lr_seq = x_lr[0]  # shape: (seq_len, h, w, c)
        lr_resized = tf.image.resize(x_lr_seq, Config.HR_SHAPE, method="bilinear") # shape: (seq_len, HR_h, HR_w, c)
        
        pred_bilinear = lr_resized[target_seq_idx, :, :, 0].numpy() * std_hr + mean_hr
        np.save(os.path.join(out_dir, f"fullframe_PUB_BASELINE_BILINEAR.npy"), pred_bilinear)
        
        # Save ground truth
        truth = y_true[0, target_seq_idx, :, :, 0].numpy() * std_hr + mean_hr
        np.save(os.path.join(out_dir, f"fullframe_PUB_TRUTH.npy"), truth)
        return
        
    models_to_test = [
        ("unet", "Ablation_UNET_Legacy_S42_best.h5", "Ablation_UNET"),
        ("unet", "Tiles_UNET_S42_best.h5", "Tiles_UNET"),
        ("convlstm", "Ablation_LSTM_Legacy_S42_best.h5", "Ablation_LSTM"),
        ("convlstm", "Tiles_LSTM_S42_best.h5", "Tiles_LSTM"),
    ]
    
    for mtype, mfile, mdesc in models_to_test:
        path = os.path.join(PROJECT_ROOT, "experiments", "models", mfile)
        if not os.path.exists(path):
            print(f"Skipping {mdesc}, {path} not found.")
            continue
            
        print(f"Generating fullframe pred for {mdesc}...")
        model = build_model(mtype)
        model.load_weights(path)
        
        y_pred = model((x_lr, x_st), training=False)
        pred_frame = y_pred[0, target_seq_idx, :, :, 0].numpy() * std_hr + mean_hr
        
        np.save(os.path.join(out_dir, f"fullframe_PUB_{mdesc}.npy"), pred_frame)

if __name__ == "__main__":
    main()
