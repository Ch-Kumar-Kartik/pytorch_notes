import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class WineDataset(Dataset):
    # Initialize data, download, etc.
    # read with numpy or pandas
    def __init__(self):
        xy = np.loadtxt('./09_jwine.csv', dtype = np.float32, skiprows = 1)
        self.n_samples = xy.shape[0]
        
        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
        
dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True, num_workers = 2)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epochs):
    for i, (input, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1)%5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

# train_dataset = torchvision.datasets.MNIST(root='./data', 
#                                            train=True, 
#                                            transform=torchvision.transforms.ToTensor(),  
#                                            download=True)

# train_loader = DataLoader(dataset=train_dataset, 
#                                            batch_size=3, 
#                                            shuffle=True)

# # look at one random sample
# dataiter = iter(train_loader)
# data = next(dataiter)
# inputs, targets = data
# print(inputs.shape, targets.shape)