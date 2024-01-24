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
T = np.fromfile("T.dat", sep=" ").reshape(
    (myphi.layers, myphi.d_c, myphi.d_c))
b = np.fromfile("b.dat", sep=" ").reshape(
    (myphi.layers, myphi.d_c))
e = np.fromfile("e.dat", sep=" ").reshape(
    (myphi.layers, myphi.d_c, len(myphi.basis)))
myphi.set_weights(T, b, e)

# assembling the a matrix from data
gamma = 1.0e-2
A = np.fromfile("A.dat", sep=" ")
rhs = np.fromfile("rhs.dat", sep=" ")

c = np.linalg.solve(A + gamma/nsamples*np.eye(myphi.d_c),
                    rhs) 
myphi.set_c_gather(c)

Vic = FunctionSpace(mesh, "DG", 0)
pcg = PCG64(seed=987654321)
rg = RandomGenerator(pcg)
du = TrialFunction(V)
v = TestFunction(V)
f_in = Function(V, name="in")


# generate initial data
ic = rg.normal(Vic)
# lengthscale over which to smooth
alpha = Constant(0.05)
area = assemble(1*dx(domain=ic.ufl_domain()))
a = (alpha**2 * du.dx(0) * v.dx(0) + du * v) * dx
L = (ic / sqrt(area)) * v * dx
solve(a == L, f_in, solver_parameters={'ksp_type': 'preonly',
                                       'pc_type': 'lu'})
f_in.interpolate(Constant(1/a)*ln(1 + exp(Constant(a)*f_in)))
f_out = Function(V, name="out")
myphi.apply(f_in, f_out)

file0 = File("fake.pvd")
file0.write(f_in, f_out)
