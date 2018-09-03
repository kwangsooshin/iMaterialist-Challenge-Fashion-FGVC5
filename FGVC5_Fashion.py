import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import pandas as pd
import os


class FashionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, loader=default_loader):
        self.csv_file = pd.read_table(csv_file, sep=',', dtype={'ID': object})
        self.image_dir = image_dir
        self.loader = loader
        self.transform = transform

    def __len__(self):
        n, _ = self.csv_file.shape
        return n

    def __getitem__(self, index):
        row = self.csv_file.iloc[index]
        image_path = os.path.join(self.image_dir, row[0] + ".jpg")
        image = self.loader(image_path)
        target = np.asarray(row[1:].values.tolist(), dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return image, target