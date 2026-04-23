import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)