"""
Loss functions shared across TensorFlow and PyTorch training code.
"""

# -----------------------------
# TensorFlow utilities
# -----------------------------

try:
    import tensorflow as tf
except ImportError:
    tf = None

def tf_hybrid_loss(alpha=0.8, max_val=1.0):
    """
    Hybrid loss: (1-alpha)*MSE + alpha*SSIM.
    Works with 4D (B,H,W,C) or 5D (B,T,H,W,C).
    """
    if tf is None:
        raise ImportError("TensorFlow no está disponible.")
    
    def _flatten_time(x):
        if x.shape.rank == 5:
            s = tf.shape(x)
            x = tf.reshape(x, (s[0] * s[1], s[2], s[3], s[4]))
        return x
    
    def loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
        y_true_flat = tf.cast(_flatten_time(y_true), tf.float32)
        y_pred_flat = tf.cast(_flatten_time(y_pred), tf.float32)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=max_val))
        return (1 - alpha) * mse_loss + alpha * ssim_loss
    
    return loss

# -----------------------------
# PyTorch utilities
# -----------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    def _gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(1, 1, 1, -1) if size == 1 else g.reshape(1, 1, size, 1)

    def _create_window(window_size, channel, device, dtype):
        _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device=device, dtype=dtype)

    def ssim_torch(img1, img2, window_size=11, size_average=True):
        """
        SSIM for torch tensors.
        img1/img2: (B, C, H, W)
        """
        channel = img1.size(1)
        window = _create_window(window_size, channel, img1.device, img1.dtype)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        return ssim_map.mean(1).mean(1).mean(1)

    class TorchHybridLoss(nn.Module):
        """
        Hybrid loss: (1-alpha)*MSE + alpha*SSIM for PyTorch.
        """
        def __init__(self, alpha=0.8, window_size=11):
            super().__init__()
            self.alpha = alpha
            self.window_size = window_size

        def forward(self, pred, target):
            # pred/target: (B, T, C, H, W) o (B, C, H, W)
            if pred.dim() == 5:
                _, _, c, h, w = pred.shape
                pred_flat = pred.view(-1, c, h, w)
                target_flat = target.view(-1, c, h, w)
            else:
                pred_flat = pred
                target_flat = target

            mse_loss = F.mse_loss(pred, target)
            ssim_val = ssim_torch(pred_flat, target_flat, window_size=self.window_size)
            ssim_loss = 1 - ssim_val

            return (1 - self.alpha) * mse_loss + self.alpha * ssim_loss
else:
    def ssim_torch(*_args, **_kwargs):
        raise ImportError("PyTorch no está disponible.")

    class TorchHybridLoss:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch no está disponible.")
