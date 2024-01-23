from firedrake import *
import numpy as np
from firedrake.petsc import PETSc

class phi(object):
    def __init__(self, mesh, V, d_c, l):
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
                                      d_c)
        self.d_c = d_c
        self.layers = layers
        self.c_gather = None
        self.basis = None
        self.T = None
        self.b = None
        self.e = None
        
    def apply(self, f_in):
        """
        Apply the map to f_in, returning result in f_out
        """
        f_layer = self.scatter(f_in)
        for layer in self.layers:
            f_layer = self.nonlocal_layer(f_layer, layer)
        f_out = self.gather(f_layer)
        return f_out

    def set_weights(self, T, b, e):
        assert self.basis, "basis is not set."
        assert T.shape == (self.layers, self.d_c. self.d_c)
        assert b.shape == (self.layers, self.d_c)
        assert e.shape == (self.layers, self.d_c, len(self.basis))
        self.T = T
        self.b = b
        self.e = e
    
    def non_local_layer(f_in, l):
        """
        Apply one nonlocal layer to f_in [from Vv]
        return output in f_out [from Vv]
        l is the layer index
        """
        assert self.T, "T is not set."
        assert self.b, "b is not set."
        assert self.e, "e is not set."
        assert self.basis, "basis is not set."

        # Lg = sigma(Tg + b + sum_j <e_j.g, phi_j> * phi_j)
        
        f_out = Function(self.Vv)
        f_outs = f_out.subfunctions
        f_ins = f_in.subfunctions
        for i in range(self.d_c):
            # local part
            f_out[i] += Constant(self.b[l, i])
            for j in range(self.d_c):
                f_outs[i] += Constant(self.T[l, i, j])*f_ins[j]
            # nonlocal part
            for k in range(self.basis.len):
                for j in range(self.d_c):
                    coeff = assemble(inner(f_ins[j], self.basis[k])*dx)
                    f_outs[i] += Constant(coeff*
                                          self.e[l, j, k])*self.basis[k]
            # sigma
            f_out[i].assign(self.sigma(f_out[i]))
        # protect against memory blowouts
        PETSc.garbage_cleanup(PETSc.COMM_SELF)
        return f_out

    def sigma(self, f):
        #  here we use the softplus
        x = Constant(self.scale)*f
        return log(1 + exp(x))  # returns an expression

    def set_c_gather(c):
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
        f_out = Function(Vv)
        f_outs = f_out.subfunctions
        for i in range(self.d_c):
            f_outs[i].assign(f_out)
        return f_out
