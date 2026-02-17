"""
Runtime Config selector.
Chooses GPU config when running on a server (CUDA env) or when explicitly requested.
"""

import os
from .config import Config as BaseConfig
from .gpu_server_config import GPUServerConfig as GPUConfig


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def use_gpu_config() -> bool:
    if _truthy(os.getenv("USE_GPU_CONFIG", "")):
        return True
    if _truthy(os.getenv("GPU_SERVER", "")):
        return True
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and cuda_visible != "-1":
        return True
    nvidia_visible = os.getenv("NVIDIA_VISIBLE_DEVICES", "")
    if nvidia_visible and nvidia_visible not in {"", "void"}:
        return True
    return False


Config = GPUConfig if use_gpu_config() else BaseConfig


def get_config():
    return Config
