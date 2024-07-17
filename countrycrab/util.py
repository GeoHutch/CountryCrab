from copy import deepcopy
from itertools import product
from numbers import Number

import numpy as np
import scipy.sparse

class NDSparseMatrix:
    """
    Utility class --- a growable, sparse tensor of arbitrary rank
    stored as a hash map from indices (tuples) to values.
    """
    # AF: tehwalrus on StackOverflow (https://stackoverflow.com/questions/7685128/sparse-3d-matrix-array-in-python)
    def __init__(self, rank, default=0.0):
        self.rank = rank
        self.default = default
        self.elements = {}
        self._shape = np.array([0 for _ in range(rank)])

    def __setitem__(self, idx, value):
        if isinstance(idx, Number):
            idx = (idx,)
        self.elements[idx] = value
        self._shape = np.maximum(self._shape, [ii+1 for ii in idx])

    def __getitem__(self, idx):
        if isinstance(idx, Number):
            idx = (idx,)
        try:
            value = self.elements[idx]
        except KeyError:
            value = self.default
        return value
    
    def __add__(self, other):
        assert self.rank == other.rank
        assert self.default == other.default
        sum_tensor = self.__class__(self.rank, self.default)
        for (idx, val) in self.elements.items():
            sum_tensor[idx] = val
        for (idx, val) in other.elements.items():
            sum_tensor[idx] += val - self.default
        return sum_tensor
    
    def __truediv__(self, other_scalar):
        assert isinstance(other_scalar, Number)
        o = self.__class__(self.rank, self.default/other_scalar)
        for k,v in self.iternz():
            o[k] = v/other_scalar
        return o


    def shape(self):
        return tuple(self._shape)
    
    def todense(self):
        o = np.zeros(self.shape())
        o += self.default
        for (k,v) in self.elements.items():
            o[k] = v
        return o
    
    def tolil(self):
        assert self.default == 0
        assert self.rank == 2
        lil = scipy.sparse.lil_matrix(self.shape())
        for (k,v) in self.iternz():
            lil[k] = v
        return lil
    
    def tocsc(self):
        return scipy.sparse.csc_matrix(self.tolil())
    
    def transpose(self, axorder):
        transposed = self.__class__(self.rank, self.default)
        transposed_index = lambda idx: tuple([idx[ax] for ax in axorder])
        for (idx, nzv) in self.elements.items():
            transposed.elements[transposed_index(idx)] = nzv
        return transposed

    def iternz(self):
        yield from self.elements.items()

class DoPMultivariateMonomial:
    def __init__(self, powers):
        self.powers = powers

    def __mul__(self, right):
        n = deepcopy(self.powers)
        for (v,p) in right.powers.items():
            if v in n.keys():
                n[v] += p
            else:
                n[v] = p
        return DoPMultivariateMonomial(n)

    def degree(self):
        return sum(self.powers.values())
    
    def is_multilinear(self):
        return not any([v != 1 for v in self.powers.values()])

    def to_index(self):
        return tuple(self.powers.keys())



class DoCMultivariatePolynomial:
    def __init__(self, vals):
        self.vals = vals

    def __mul__(self, right):
        n = {}
        for ((kl, vl), (kr, vr)) in product(self.vals.items(), right.vals.items()):
            nk, nv = kl*kr, vl*vr
            if nk in n.keys():
                n[nk] += nv
            else:
                n[nk] = nv
        return DoCMultivariatePolynomial(n)
    
    def __add__(self, right):
        n = deepcopy(self.vals)
        for (mr,cr) in right.vals.items():
            if mr in n.keys():
                n[mr] += cr
            else:
                n[mr] = cr
        return DoCMultivariatePolynomial(n)
    
    def terms(self):
        yield from self.vals.items()

    def push_coeffs_to_tensors(self, tensors):
        for (monomial, coeff) in self.vals.items():
            if not monomial.is_multilinear():
                raise RuntimeError("monomial is not multilinear")
            tensors[monomial.degree()][monomial.to_index()] += coeff

    @classmethod
    def multiplicative_identity(cls):
        """
        Returns the multiplicative identity element for the field of DoCMPs
        has one trivial monomial with coefficient unity.
        """
        return cls({DoPMultivariateMonomial({}): 1})

    @classmethod
    def additive_identity(cls):
        return cls({})
    

