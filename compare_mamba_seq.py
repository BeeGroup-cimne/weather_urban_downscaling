import os
import sys
import pandas as pd

PROJECT_ROOT = "/app"
sys.path.append(PROJECT_ROOT)

def compare_logs():
    log_dir = os.path.join(PROJECT_ROOT, "experiments", "logs")
    log_6 = os.path.join(log_dir, "Tiles_MAMBA_S42_log.csv")
    log_12 = os.path.join(log_dir, "Ablation_MAMBA_Legacy_S42_SEQ12_log.csv")

    res = []
    
    if os.path.exists(log_6):
        df6 = pd.read_csv(log_6)
        res.append({
            "Seq_Len": 6, 
            "Best_Val_MAE": df6["val_mae"].min(), 
            "Best_Val_MSE": df6["val_loss"].min() if "val_loss" in df6.columns else df6["val_mse"].min()
        })
    else:
        print("⚠️ No se encontró el log de MAMBA SEQ=6")

    if os.path.exists(log_12):
        df12 = pd.read_csv(log_12)
        res.append({
            "Seq_Len": 12, 
            "Best_Val_MAE": df12["val_mae"].min(), 
            "Best_Val_MSE": df12["val_loss"].min() if "val_loss" in df12.columns else df12["val_mse"].min()
        })
    else:
        print("⚠️ No se encontró el log de MAMBA SEQ=12")

    if res:
        print("\n📊 COMPARATIVA MAMBA (SEQ=6 vs SEQ=12)")
        print(pd.DataFrame(res).to_string(index=False))

if __name__ == "__main__":
    compare_logs()
