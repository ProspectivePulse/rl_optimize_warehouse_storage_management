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

import yaml
import os

def load_config(filename="config.yaml"):
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.join(root_path, filename)
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

