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

# --- SSIM Implementation for PyTorch (Matches tf.image.ssim) ---
def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.reshape(1, 1, 1, -1) if size == 1 else g.reshape(1, 1, size, 1) # Simplified 1D guassian

def create_window(window_size, channel):
    # Create 2D gaussian window
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_torch(img1, img2, window_size=11, size_average=True):
    # Assumes img1, img2 are (Batch, Channel, Height, Width) or (Batch*Time, C, H, W)
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.84): # Matches TF default approx
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # pred/target: (Batch, Time, H, W) -> flatten Time to SSIM works on frames
        # Wait, model output is (Batch, Time, 1, H, W).
        
        b, t, c, h, w = pred.shape
        pred_flat = pred.view(-1, c, h, w)
        target_flat = target.view(-1, c, h, w)
        
        mse_loss = self.mse(pred, target)
        
        # SSIM expects normalized images roughly [0,1] or similar scale.
        # But TF code uses max_val=5.0.
        # We'll rely on the standard implementation.
        ssim_val = ssim_torch(pred_flat, target_flat, window_size=11)
        ssim_loss = 1 - ssim_val
        
        # TF Alpha was 0.8
        # TF Alpha was 0.8
        return (1 - self.alpha) * mse_loss + self.alpha * ssim_loss

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
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # 3. Model
    model = DownsrUNetMamba().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = HybridLoss().to(device)
    
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
