# just checking the phi methods execute
from firedrake import *
import numpy as np
from phi import phi

mesh = UnitIntervalMesh(50)
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

# setting the c_gather [we should solve for this in general but here
# we just randomise it to get numbers]

c = np.random.randn(myphi.d_c)
myphi.set_c_gather(c)

f_in = Function(V).interpolate(exp(sin(2*pi*x)) + cos(2*pi*x)**3)
f_out = Function(V)
myphi.apply(f_in, f_out)

# testing the assembly to see if it executes
# we'll just feed f_in and f_out back in which is nonsense
# but enough to check code runs

A = np.zeros((myphi.d_c, myphi.d_c))
b = np.zeros((myphi.d_c,))

myphi.increment_ls_system(A, b, (f_in, f_out))
