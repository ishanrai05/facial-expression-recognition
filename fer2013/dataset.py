from utils import arr_to_img
import numpy as np
from PIL import Image
import torch


# Define a pytorch dataloader for this dataset
class FRE2013(torch.utils.data.Dataset):
    def __init__(self, df, width, height, transform=None):
        self.width = width
        self.height = height
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = arr_to_img(self.df['pixels'].iloc[index], self.width, self.height)
        X = X[:, :, np.newaxis]
        X = np.concatenate((X, X, X), axis=2)        
        X = Image.fromarray(X)
        y = torch.tensor(int(self.df['emotion'].iloc[index]))
        if self.transform:
            X = self.transform(X)

        return X, y