from firedrake import *
import numpy as np
from firedrake.petsc import PETSc

class phi(object):
    def __init__(self, mesh, V, d_c, layers):
        """
        Class for nonlocal functional random feature maps

        inputs: 
        V : Function space for input/output (cannot be mixed or VFS)
        d_c : channel bandwidth
        layers : number of nonlocal layers
        """
        self.V = V
        self.Vv = VectorFunctionSpace(V.mesh(),
                                      V.ufl_element(),
                                      dim=d_c)
        self.d_c = d_c
        self.layers = layers
        self.c_gather = None
        self.basis = None
        self.T = None
        self.b = None
        self.e = None
        self.scale = 1.0
        self.f_layer = Function(self.Vv)
        self.f_layer1 = Function(self.Vv)

    def run(self, f_in):
        """
        Apply the map up to final layer to f_in, 
        placing result in self.f_layer
        """
        f_layer = self.f_layer
        f_layer1 = self.f_layer1
        self.scatter(f_in, f_layer)
        for layer in range(self.layers):
            self.nonlocal_layer(f_layer, f_layer1, layer)
            f_layer.assign(f_layer1)


    def apply(self, f_in, f_out):
        """
        Apply full map to f_in, placing result in f_out
        """
        f_layer = self.f_layer
        self.run(f_in)
        self.gather(f_layer, f_out)
        return f_out

    def increment_ls_system(self, mat, rhs, pair):
        """
        Add entries to an existing least squares matrix
        and rhs
        for obtaining c_gather coming from a pair (X, Y)
        of inputs and outputs

        mat - a (d_c, d_c) numpy array
        rhs - a (d_c,) numpy array
        pair - a size 2 tuple containing input and output functions
        """

        self.run(pair[0])
        f_layer = self.f_layer
        for i in range(self.d_c):
            for j in range(self.d_c):
                mat[i, j] += assemble(
                    inner(f_layer.sub(i), f_layer.sub(j))*dx)
            rhs[i] += assemble(inner(f_layer.sub(i), pair[1])*dx)

    def set_basis(self, basis):
        self.basis = basis

    def set_weights(self, T, b, e):
        assert self.basis, "basis is not set."
        assert T.shape == (self.layers, self.d_c, self.d_c)
        assert b.shape == (self.layers, self.d_c)
        assert e.shape == (self.layers, self.d_c, len(self.basis))
        self.T = T
        self.b = b
        self.e = e

    def nonlocal_layer(self, f_in, f_out, l):
        """
        Apply one nonlocal layer to f_in [from Vv]
        return output in f_out [from Vv]
        l is the layer index
        """
        assert self.T.all(), "T is not set."
        assert self.b.all(), "b is not set."
        assert self.e.all(), "e is not set."
        assert self.basis, "basis is not set."

        # Lg = sigma(Tg + b + sum_j <e_j.g, phi_j> * phi_j)
        
        f_out.assign(0.)

        f_exp = []
        for i in range(self.d_c):
            # local part
            f_out.sub(i).assign(f_in.sub(i) + Constant(self.b[l, i]))
            for j in range(self.d_c):
                f_out.sub(i).assign(f_out.sub(i) +
                                    Constant(self.T[l, i, j])*f_in.sub(j))
            # nonlocal part
            for k, basis_fn in enumerate(self.basis):
                for j in range(self.d_c):
                    coeff = assemble(inner(f_in.sub(j), basis_fn)*dx)
                    f_out.sub(i).assign(f_out.sub(i) +
                                        Constant(coeff*
                                                 self.e[l, j, k])*basis_fn)
            # sigma
            self.sigma(f_out)
        # protect against memory blowouts
        # PETSc.garbage_cleanup(PETSc.COMM_SELF)
        return f_out

    def sigma(self, f):
        #  here we use the softplus
        expr = []
        for i in range(self.d_c):
            x = Constant(self.scale)*f.sub(i)
            expr.append(ln(1 + exp(x)))
        f.interpolate(as_vector(expr))

    def set_c_gather(self, c):
        assert c.shape == (self.d_c,)
        self.c_gather = c
    
    def gather(self, f_in, f_out):
        """
        given f_in from VFS, 
        construct f_out = sum_i c_i f_in[i]
        where c_i are coefficients in self.c_gather
        """
        assert self.c_gather.all(), "c_gather is not set."
        f_out.assign(0.)
        for i in range(self.d_c):
            f_out.assign(f_out
                         + Constant(self.c_gather[i])*f_in.sub(i))

    def scatter(self, f_in, f_out):
        """
        Scatter f_in out to a channel of width d_c
        returns: f_out, the result
        """
        f_out.assign(0.)
        f_splat = []
        for i in range(self.d_c):
            f_splat.append(f_in)
        f_out.interpolate(as_vector(f_splat))
