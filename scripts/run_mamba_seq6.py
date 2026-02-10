#!/usr/bin/env python3
"""
Run Mamba training with SEQ_LEN=6 using the existing torch training pipeline.
This is intended for paper experiments (baseline temporal context).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config.runtime import Config
from scripts.torch_gpu_train import train_gpu


def apply_overrides():
    Config.SEQ_LEN = 6
    Config.EFFECTIVE_BATCH_SIZE = Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS
    print("✅ Mamba overrides:")
    print(f"   SEQ_LEN={Config.SEQ_LEN}")
    print(f"   BATCH_SIZE={Config.BATCH_SIZE}")
    print(f"   GRAD_ACC={Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   EFFECTIVE_BATCH={Config.EFFECTIVE_BATCH_SIZE}")


if __name__ == "__main__":
    apply_overrides()
    train_gpu()
