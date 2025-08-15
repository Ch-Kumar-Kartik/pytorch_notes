import torch

x = torch.tensor(1.0) # input
y = torch.tensor(2.0) # ground truth

w = torch.tensor(1.0, requires_grad = True) # weights

# doing a forward pass and computing the loss
y_hat = w * x 
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)

