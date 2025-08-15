import torch 

x = torch.randn(3, requires_grad = True) # Setting requires_grad=True means PyTorch will track operations on x to compute gradients
# a computation graph consisting of tensors which have requires_grad = True
print(x)

y = x + 2 # Computes y = [x_0 + 2, x_1 + 2, x_2 + 2] element-wise. This is part of the computation graph.
print(y)

z = y*y*2 # Computes z = 2 * y^2 = [2 * (x_0 + 2)^2, 2 * (x_1 + 2)^2, 2 * (x_2 + 2)^2]. The tensor z has shape (3,)
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32) # used as the gradient weights for the vector-Jacobian product
z.backward(v) # vector jacobian product (will calculate gradient i.e dz/dx) (also if z is a scalar it doesn't require any argument and if its a )
print(x.grad) # will store the gradients and will only work if requires_grad = True

'''
.backward() computes gradients of a scalar loss with respect to tensors in computation_graph that have requires_grad = True and these gradients are stored in .grad attribute of the input feature

When the tensor (e.g., z in your code) is a scalar 
(a single value), .backward() can implicitly assume 
the gradient of the scalar with respect to itself 
is 1, and it computes gradients for all preceding 
tensors (like x) in the computation graph. 
For example, if z = z.mean() were uncommented in 
your code, z would be a scalar, and z.backward() 
would work without arguments.

z = [z_0, z_1, z_2] and input x = [x_0, x_1, x_2], the Jacobian is a matrix of shape (3, 3):
$$J = \begin{bmatrix}
\frac{\partial z_0}{\partial x_0} & \frac{\partial z_0}{\partial x_1} & \frac{\partial z_0}{\partial x_2} \\
\frac{\partial z_1}{\partial x_0} & \frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} \\
\frac{\partial z_2}{\partial x_0} & \frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2}
\end{bmatrix}$$

To compute gradients, PyTorch needs to reduce z to a 
scalar. This is done by computing a weighted sum of 
z’s components, weighted by a vector v (the 
argument to .backward(v)). The scalar is:
    v0z0 + v1z1 + v2z2 = v'z (where v' is v transpose)
    
The gradient of s with respect to x is the 
vector-Jacobian product:
$$\frac{\partial s}{\partial x} = v^T J = \left[ \sum_i v_i \frac{\partial z_i}{\partial x_0}, \sum_i v_i \frac{\partial z_i}{\partial x_1}, \sum_i v_i \frac{\partial z_i}{\partial x_2} \right]$$

=> put the below code in latex to visualize :
To compute x.grad, we need to derive the gradients mathematically:

Let x = [x_0, x_1, x_2], so y = [x_0 + 2, x_1 + 2, x_2 + 2], and z = [2 * (x_0 + 2)^2, 2 * (x_1 + 2)^2, 2 * (x_2 + 2)^2].
The scalar s is:
$$s = v_0 z_0 + v_1 z_1 + v_2 z_2 = v_0 \cdot 2 (x_0 + 2)^2 + v_1 \cdot 2 (x_1 + 2)^2 + v_2 \cdot 2 (x_2 + 2)^2$$

We need the gradient $\frac{\partial s}{\partial x_i}$ for each $x_i$. For each component:
$$z_i = 2 (x_i + 2)^2$$
$$\frac{\partial z_i}{\partial x_i} = 2 \cdot 2 (x_i + 2) \cdot 1 = 4 (x_i + 2)$$

Since $s = \sum v_i z_i$, the gradient is:
$$\frac{\partial s}{\partial x_i} = v_i \cdot \frac{\partial z_i}{\partial x_i} = v_i \cdot 4 (x_i + 2) = 4 v_i (x_i + 2)$$

With v = [0.1, 1.0, 0.001], the gradients are:
$$x.grad = [4 \cdot 0.1 \cdot (x_0 + 2), 4 \cdot 1.0 \cdot (x_1 + 2), 4 \cdot 0.001 \cdot (x_2 + 2)] = [0.4 (x_0 + 2), 4 (x_1 + 2), 0.004 (x_2 + 2)]$$
'''

# preventing gradient history
# during training loop when we want to update weights then requires_grad should not be a part of the gradient computation
# option 1 is to : x.requires_grad_(False)
# option 2 is x.detach()
# option 3 is with torch.no_grad()

x.requires_grad_(False) # trailing underscore means in place operation
print(x)

y = x.detach()

with torch.no_grad():
    y = x + 2
    print(y)
    
    