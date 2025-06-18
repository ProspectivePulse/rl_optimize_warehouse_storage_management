import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def log_to_tensorboard(log_dir="logs/tensorboard"):
    return SummaryWriter(log_dir)
