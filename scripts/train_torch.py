import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import xarray as xr
import sys
import os
import torch.nn.functional as F

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.torch_engine.model_mamba import DownsrUNetMamba
from src.data_loader import BigDataPipeline
from config.config import Config
from src.losses import TorchHybridLoss


# --- Zarr Dataset ---
class ZarrIterableDataset(IterableDataset):
    def __init__(self, cache_dir, config, split='train'):
        super().__init__()
        self.cache_dir = cache_dir
        self.cfg = config
        self.ds = xr.open_zarr(cache_dir, consolidated=True)
        
        # Detect dims
        self.lr_lat = next((d for d in self.ds['lr_input'].dims if d in ['latitude', 'lat', 'y']), 'y')
        self.lr_lon = next((d for d in self.ds['lr_input'].dims if d in ['longitude', 'lon', 'x']), 'x')
        self.lr_time = next((d for d in self.ds['lr_input'].dims if d in ['time', 'valid_time', 't']), 'time')
        self.lr_var = next((d for d in self.ds['lr_input'].dims if d in ['variable', 'channel', 'var']), None)
        
        self.hr_y = next((d for d in self.ds['hr_target'].dims if d in ['y', 'latitude', 'lat']), 'y')
        self.hr_x = next((d for d in self.ds['hr_target'].dims if d in ['x', 'longitude', 'lon']), 'x')

        total_len = self.ds.dims[self.lr_time]
        split_idx = int(total_len * self.cfg.SPLIT_FRACTION)
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = split_idx
        else:
            self.start_idx = split_idx
            self.end_idx = total_len
            
        print(f"   📂 Dataset ({split}) range: {self.start_idx} -> {self.end_idx}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = self.start_idx
        end = self.end_idx

        if worker_info is not None:
            per_worker = int(np.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = start + worker_id * per_worker
            end = min(start + per_worker, end)
        
        # Simple slicing for now (assumes single worker or we replicate data)
        # Ideally split workload if num_workers > 0
        
        seq_len = self.cfg.SEQ_LEN
        
        # Precompute static
        # Logic copied from BigDataPipeline but adapted for quick access
        # Assuming static data is already generated and saved in cache or we create dummy logic
        # Ideally we should read `processed_cache_zarr/static_processed.npy`
        static_path = os.path.join(PROJECT_ROOT, "processed_cache_zarr/static_processed.npy")
        if os.path.exists(static_path):
            static_data = np.load(static_path)
        else:
            # Fallback
            static_data = np.zeros((*self.cfg.HR_SHAPE, self.cfg.STATIC_CHANNELS), dtype='float32')
            
        if static_data.ndim == 2: static_data = static_data[..., np.newaxis]
        mean_st = np.mean(static_data, axis=(0, 1), keepdims=True)
        std_st = np.std(static_data, axis=(0, 1), keepdims=True)
        static_norm = (static_data - mean_st) / (std_st + 1e-6)
        
        # Iterate
        da_lr = self.ds['lr_input']
        da_hr = self.ds['hr_target']
        
        try:
            for i in range(start, end - seq_len):
                # 1. LR Input
                # xarray uses dimension names
                if self.lr_var:
                    # (Time, Lat, Lon, Var)
                    x_lr = da_lr.isel({self.lr_time: slice(i, i+seq_len)}) \
                                .transpose(self.lr_time, self.lr_lat, self.lr_lon, self.lr_var) \
                                .values
                else:
                    x_lr = da_lr.isel({self.lr_time: slice(i, i+seq_len)}) \
                                .transpose(self.lr_time, self.lr_lat, self.lr_lon) \
                                .values[..., np.newaxis]
                
                # Permute for PyTorch: (Time, Lat, Lon, Chan) -> (Time, Chan, Lat, Lon)
                # But our model expects (Batch, Time, Chan, H, W)
                # So we yield (Time, Chan, Lat, Lon)
                x_lr = np.moveaxis(x_lr, -1, 1) # (T, C, H, W)
                
                # 2. HR Target
                y_hr = da_hr.isel({self.lr_time: slice(i, i+seq_len)}) \
                            .transpose(self.lr_time, self.hr_y, self.hr_x) \
                            .values[..., np.newaxis]
                
                y_hr = np.moveaxis(y_hr, -1, 1) # (T, 1, H, W)
                
                # 3. Static
                # TODO: This consumes high RAM. Check if model supports broadcasting to avoid np.repeat.
                x_st = np.repeat(static_norm[np.newaxis, ...], seq_len, axis=0) # (T, H, W, C)
                x_st = np.moveaxis(x_st, -1, 1) # (T, C, H, W)
                
                yield torch.from_numpy(x_lr).float(), torch.from_numpy(x_st).float(), torch.from_numpy(y_hr).float()
                
        except Exception as e:
            print(f"⚠️ Error in dataloader: {e}")
            return

def train():
    print("🚀 Iniciando Entrenamiento Mamba (PyTorch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚙️ Dispositivo: {device}")
    
    # 1. Pipeline & Generation
    print("📦 Inicializando Big Data Pipeline...")
    pipeline = BigDataPipeline(Config)
    pipeline.process_static_data()
    pipeline.run_etl_process()
    
    # 2. Datasets
    train_dataset = ZarrIterableDataset(Config.PATH_CACHE, Config, split='train')
    val_dataset = ZarrIterableDataset(Config.PATH_CACHE, Config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # 3. Model
    model = DownsrUNetMamba().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = TorchHybridLoss(alpha=0.84).to(device)
    
    # Test Forward Pass
    print("🧪 Verificando pasang Forward...")
    try:
        dummy_lr, dummy_st, dummy_tg = next(iter(train_loader))
        dummy_lr = dummy_lr.to(device)
        dummy_st = dummy_st.to(device)
        out = model(dummy_lr, dummy_st)
        print(f"   ✅ Output Shape: {out.shape}")
    except StopIteration:
        print("❌ Dataset vacío o error de lectura.")
        return

    # 4. Loop
    EPOCHS = 3 # Demo run
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        steps = 0
        
        for batch_idx, (lr, st, target) in enumerate(train_loader):
            lr, st, target = lr.to(device), st.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(lr, st)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f"   Epoch {epoch+1} [Batch {steps}] Loss: {loss.item():.4f}", end='\r')
                
        print(f"Epoca {epoch+1} | Train Loss: {train_loss/steps:.6f}")
        
    print("✅ Entrenamiento PyTorch finalizado.")

if __name__ == "__main__":
    train()
