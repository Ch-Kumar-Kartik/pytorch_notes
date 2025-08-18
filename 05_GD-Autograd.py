# # numpy implementation
# import numpy as np

# # f = w * x
# X = np.array([1, 2, 3, 4], dtype = np.float32)
# Y = np.array([2,4, 6, 8], dtype = np.float32)

# w = 0.0

# # model prediction
# def forward(x):
#     return w*X

# # loss = MSE
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

# # gradient 
# # MSE = 1/N * (w*x - y)**2
# # dJ/dw = 1/N 2x (w*x - y)
# def gradient(x, y, y_predicted):
#     return np.dot(2*x, y_predicted - y).mean()

# print(f'Prediction before training: f(5)')

# # Training
# learning_rate = 0.01
# n_iters = 10

# for epoch in range(n_iters):
#     # prediction = forward pass
#     y_pred = forward(X)
    
#     # loss
#     l = loss(Y, y_pred)
    
#     # gradients
#     dw = gradient(X, Y, y_pred)
    
#     # update weights 
#     w -= learning_rate * dw
    
#     if epoch % 1 == 0:
#         print(f'Prediction after training: f(5) = {forward(5):.3f}')

# torch implementation
import torch

# f = w * x
X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2,4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32)

# model prediction
def forward(x):
    return w*X

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

print(f'Prediction before training: f(5)')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights 
    with torch.no_grad(): # torch.no_grad() is used to prevent PyTorch from tracking the update operation, as it’s not part of the loss computation.
        w -= learning_rate * w.grad
        
    # zero gradients 
    w.grad.zero_()
    
    # Without this, the next iteration’s gradient would add to w.grad = -4.0, leading to incorrect updates.
    
    '''
    Without w.grad.zero_(), the new gradient (-9.5) is added to the existing w.grad = -10.0:
    w.grad = w.grad + (new_w_grad) = -19.5
    '''
    if epoch % 1 == 0:
        print(f'Prediction after training: f(5) = {forward(5):.3f}')
