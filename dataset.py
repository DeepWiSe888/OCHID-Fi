import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import  gc
from config import *


class RFDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """

    def __init__(self,lines,num_train, path_x, path_y):
        super().__init__()
        # list of files
        self.lines = lines
        self.len_ = num_train
        self.path_x = path_x
        self.path_y = path_y

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        
        name = self.lines[index].split('\n')[0]

        mat = sio.loadmat(self.path_x + name)['data']

        real = mat.real
        imag = mat.imag
        real = real.reshape(10, 40, 40, 1)
        imag = imag.reshape(10, 40, 40, 1)
        x = np.concatenate([real, imag], axis=-1)

        y_t = sio.loadmat(self.path_y + name)['label_t']
        y_gt = sio.loadmat(self.path_y + name)['label_gt']
        phi = sio.loadmat(self.path_y + name)['phi']
        
        return x,y_t,y_gt,phi

def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)