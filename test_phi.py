# a bit of test code that just checks that the code in phi executes
from firedrake import *
import numpy as np

mesh = UnitIntervalMesh(50)
V = FunctionSpace(mesh, "CG", 1)
myphi = phi(mesh, V, d_c=10, l=5)
x, = SpatialCoordinate(mesh)

# setting the basis for the nonlocality
basis = []
basis = [b0]
kmax = 3
for k in range(1, kmax+1):
    f = Function(V).assign(sin(2*pi*k*x))
    basis.extend(f)
    f = Function(V).assign(cos(2*pi*k*x))
    basis.extend(f)
myphi.set_basis(basis)

# setting the random features
T = np.random.randn((myphi.layers, myphi.d_c, myphi.d_c))
b = np.random.randn((myphi.layers, myphi.d_c))
e = np.random.randn((myphi.layers, myphi.d_c, len(myphi.basis)))
myphi.set_weights(T, b, e)

# setting the c_gather [we should solve for this in general but here
# we just randomise it to get numbers]

c = np.random.randn((myphi.layers,))
myphi.set_c_gather(c)

f_in = Function(V).assign(exp(sin(2*pi*x)) + cos(2*pi*x)**3)
f_out = myphi.apply(f_in)
