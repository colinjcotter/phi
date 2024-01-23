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
        
    def apply(self, f_in):
        """
        Apply the map to f_in, returning result in f_out
        """
        f_layer = self.scatter(f_in)
        for layer in range(self.layers):
            f_layer = self.nonlocal_layer(f_layer, layer)
        f_out = self.gather(f_layer)
        return f_out

    def set_basis(self, basis):
        self.basis = basis

    def set_weights(self, T, b, e):
        print(type(self))
        assert self.basis, "basis is not set."
        assert T.shape == (self.layers, self.d_c, self.d_c)
        assert b.shape == (self.layers, self.d_c)
        assert e.shape == (self.layers, self.d_c, len(self.basis))
        self.T = T
        self.b = b
        self.e = e

    def nonlocal_layer(self, f_in, l):
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
        
        f_out = Function(self.Vv)

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
        PETSc.garbage_cleanup(PETSc.COMM_SELF)
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
        assert self.c_gather, "c_gather is not set."
        f_out = Function(self.V)
        f_ins = f_in.subfunctions
        for i in range(d_c):
            f_out += self.c_gather*f_ins[i]
        return f_out
        
    def scatter(self, f_in):
        """
        Scatter f_in out to a channel of width d_c
        returns: f_out, the result
        """
        f_out = Function(self.Vv)
        f_splat = []
        for i in range(self.d_c):
            f_splat.append(f_in)
        f_out.interpolate(as_vector(f_splat))
        return f_out
