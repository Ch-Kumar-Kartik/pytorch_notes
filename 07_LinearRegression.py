import torch
import torch.nn as nn 
from sklearn import datasets #type:ignore
import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore

X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)
x = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

n_iter = 100
for epoch in range(n_iter):
    y_pred = model(x)
    l = criterion(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'loss is {l.item():.8f} for epoch:{epoch}')
        
predicted = model(x).detach().numpy()
plt.plot(X_numpy, y_numpy)
plt.plot(X_numpy, predicted, 'b')
plt.show()