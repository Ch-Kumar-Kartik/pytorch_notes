import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    # Initialize data, download, etc.
    # read with numpy or pandas
    def __init__(self, transform = None):
        xy = np.loadtxt('./09_jwine.csv', dtype = np.float32, skiprows = 1)
        self.n_samples = xy.shape[0]
        
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]
        
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
    
        return self.x_data[index], self.y_data[index]
    
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __call__(self, sample):
        inputs, target = sample
        input *= self.factor
        return inputs, target

# dataset = WineDataset((transform = ToTensor()))
# first_data = dataset[0]
# features, labels = first_data

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data


