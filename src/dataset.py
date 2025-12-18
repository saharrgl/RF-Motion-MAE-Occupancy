import numpy as np
import torch
from torch.utils.data import Dataset

def compute_motion_features(rf_windows):
    mean_amp = rf_windows.mean(axis=-1)
    var_amp  = rf_windows.var(axis=-1)
    ste      = (rf_windows**2).mean(axis=-1)

    grad_mean = np.abs(np.diff(mean_amp, axis=1, prepend=mean_amp[:, :1]))
    grad_ste  = np.abs(np.diff(ste, axis=1, prepend=ste[:, :1]))

    return np.stack([var_amp, ste, grad_mean, grad_ste], axis=-1).astype(np.float32)


class WiSARFMotionDataset(Dataset):
    def __init__(self, X, y_occ, y_act, task="occupancy"):
        self.X = X
        self.y_occ = y_occ
        self.y_act = y_act
        self.task = task

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y_occ[idx] if self.task == "occupancy" else self.y_act[idx]
        return x, y
