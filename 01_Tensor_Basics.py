import torch  # type: ignore
import numpy as np # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"

# A tensor is a multi-dimensional array used to represent data in numerical computations, especially in machine learning and scientific computing. Tensors generalize scalars (0D), vectors (1D), and matrices (2D) to higher dimensions (3D and beyond).
# Torch tensor is Designed for deep learning, supports GPU acceleration and automatic differentiation (autograd). numpy array is for General-purpose numerical computing, no built-in GPU or gradient support
x_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# x = torch.empty(tensor_size) // creates an empty tensor of dimension tensor_size
x = torch.empty(2,2,3) # a 3d tensor of shape (2,2,3)

y = torch.rand(2,2)
y = torch.zeros(2,2)
y = torch.ones(2,2, dtype = torch.float16) # also has a dtype attribute which assigns a datatype to it say float16 then all elements inside will be float16
print(y.size()) # will give the size

x = torch.tensor([1, 3, 4, 5]) # list -> tensor

# addition and in place (functions with trailing underscores are inplace operations)
print(y, x)
print(y + x) # print(torch.add(x,y))
print(f'this is inplace: {y.add_(x)}, this is y : {y}')

# subtraction, multiplication, division
print([x*y, x/y, y-x]) # print([torch.sub(x,y), torch.mul(x,y), torch.div(x,y)])
print([y.sub_(x), y.div_(x), y.mul_(x)])
      
# slicing operation 
print(x[:,0])
print(x[1,:])
print(x[1,1].item()) # can only be used if there is a single element

# reshaping operation
z = x.view(16) 
z = x.view(-1, 8) # Reshapes x into a 2D tensor with 8 columns, where the number of rows is inferred automatically using -1

# torch tensor -> numpy array
x_numpy = x_torch.numpy()

# The tensor must be on CPU (not GPU). If on GPU, move it first
# x_torch = x_torch.cpu()  # Move to CPU if on GPU
# x_numpy = x_torch.numpy()

# Memory Sharing: The resulting NumPy array shares memory with the tensor (if contiguous). Modifying one affects the other:
# x_numpy[0, 0] = 10
# print(x_torch)  # tensor([[10., 2., 3.], [4., 5., 6.]])

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a+=1
print(a)
print(b)

# numpy array -> torch tensor
x_numpy = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# Convert to PyTorch tensor
x_torch = torch.from_numpy(x_numpy)  # Option 1: Shares memory
# OR
x_torch_copy = torch.tensor(x_numpy)  # Option 2: Copies data

# The resulting tensor is on CPU. Move to GPU if needed:
x_torch = x_torch.cuda()  # Move to GPU

# torch.from_numpy(): Shares memory with the NumPy array. Modifying one affects the other