# generate data for an initial testing of the training
from firedrake import *

ncells = 1000
mesh = PeriodicUnitIntervalMesh(ncells, name="mesh0")
Vic = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)
du = TrialFunction(V)
v = TestFunction(V)
u0 = Function(V, name="input")
un = Function(V, name="output")
unp1 = Function(V)
dt = 1.0e-2
Dt = Constant(dt)
tmax = 1.0
kappa = Constant(1.0e-4)

eqn = ((unp1 - un)*v + Dt*v*unp1*unp1.dx(0) +
       kappa*Dt*unp1.dx(0)*v.dx(0))*dx
prob = NonlinearVariationalProblem(eqn, unp1)
tsolver = NonlinearVariationalSolver(prob,
                                     solver_parameters={
                                         'ksp_type':'preonly',
                                         'pc_type':'lu'})

nsamples = 1000
vtk = True

if vtk:
    pfile = File('first.pvd')

with CheckpointFile("first.h5", 'w') as afile:
    afile.save_mesh(mesh)

for i in range(nsamples):
    print(i)
    # generate initial data
    ic = rg.normal(Vic)
    # lengthscale over which to smooth
    alpha = Constant(0.05)
    area = assemble(1*dx(domain=ic.ufl_domain()))
    a = (alpha**2 * du.dx(0) * v.dx(0) + du * v) * dx
    L = (ic / sqrt(area)) * v * dx
    solve(a == L, u0, solver_parameters={'ksp_type': 'preonly',
                                         'pc_type': 'lu'})
    a = 10
    u0.interpolate(Constant(1/a)*ln(1 + exp(Constant(a)*u0)))

    un.assign(u0)
    t = 0.
    while t < tmax - dt/2:
        t += dt
        tsolver.solve()
        un.assign(unp1)
    with CheckpointFile("first.h5", 'a') as afile:
        afile.save_function(u0, idx=i)
        afile.save_function(un, idx=i)
    if vtk:
        pfile.write(u0, un)
