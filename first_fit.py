# an initial testing of the training
from firedrake import *
import numpy as np
from phi import phi

with CheckpointFile("first.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh0")

V = FunctionSpace(mesh, "CG", 1)
myphi = phi(mesh, V, d_c=10, layers=5)
x, = SpatialCoordinate(mesh)

# setting the basis for the nonlocality
f0 = Function(V).interpolate(Constant(1.0))
basis = [f0]
kmax = 3
for k in range(1, kmax+1):
    f = Function(V).interpolate(sin(2*pi*k*x))
    basis.append(f)
    f = Function(V).interpolate(cos(2*pi*k*x))
    basis.append(f)
myphi.set_basis(basis)

# setting the random features
T = np.random.randn(myphi.layers, myphi.d_c, myphi.d_c)
b = np.random.randn(myphi.layers, myphi.d_c)
e = np.random.randn(myphi.layers, myphi.d_c, len(myphi.basis))
myphi.set_weights(T, b, e)

# assembling the a matrix from data

A = np.zeros((myphi.d_c, myphi.d_c))
b = np.zeros((myphi.d_c,))
nsamples = 20
with CheckpointFile("first.h5", 'r') as afile:
    for i in range(20):
        print(i)
        f_in = afile.load_function(mesh, "input", idx=i)
        f_out = afile.load_function(mesh, "output", idx=i)
        myphi.increment_ls_system(A, b, (f_in, f_out))

gamma = 1.0e-2

x = np.linalg.solve(A + gamma/nsamples*np.eye(myphi.d_c),
                    b) 


