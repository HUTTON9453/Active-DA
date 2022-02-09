import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def get_handler():
    return DataHandler1

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        path = self.X[index]
        x = Image.open(path).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

