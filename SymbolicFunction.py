from Expr import Expr, VectorExprInterface, VectorElementInterface
from ExprShape import (ExprShape, ScalarShape, TensorShape,
        VectorShape, ListShape)
from BasisBase import BasisBase, VectorBasisBase, ScalarBasisBase

import pytest

class SymbolicFunctionBase(Expr):

    _nextFuncID = 0

    def __init__(self, basis, name):
        super().__init__(basis.shape())
        self._name = name
        self._funcID = SymbolicFunctionBase._nextFuncID
        SymbolicFunctionBase._nextFuncID += 1
        self._basis = basis

    def basis(self):
        return self._basis

    def _sameas(self, other):
        return self._funcID == other._funcID

    def __str__(self):
        return self._name


class SymbolicVectorFunctionBase(SymbolicFunctionBase, VectorExprInterface):

    def __init__(self, basis, name):

        assert(isinstance(basis.shape(), VectorShape))

        super().__init__(basis, name)
        self._elems = []
        for i in range(basis.shape().dim):
            self._elems.append(SymbolicVectorFunctionElement(self, i))

    def __getitem__(self, i):
        return self._elems[i]


class SymbolicVectorFunctionElement(Expr, VectorElementInterface):

    def __init__(self, parent, myIndex):
        super().__init__(ScalarShape())
        super(Expr, self).__init__(parent, myIndex)


    def _sameas(self, other):
        return (self.parent().sameas(other.parent()) and
            self.index())


class TestFunctionInterface:
    def __init__(self):
        pass

class ScalarTestFunction(SymbolicFunctionBase, TestFunctionInterface):
    def __init__(self, basis, name):
        assert(isinstance(basis, ScalarBasisBase))
        super().__init__(basis, name)


class VectorTestFunction(SymbolicVectorFunctionBase, TestFunctionInterface):
    def __init__(self, basis, name):
        assert(isinstance(basis, VectorBasisBase))
        super().__init__(basis, name)



def TestFunction(basis, name):

    if isinstance(basis, VectorBasisBase):
        return VectorTestFunction(basis, name)
    elif isinstance(basis, ScalarBasisBase):
        return ScalarTestFunction(basis, name)
    else:
        raise('TestFunction expected scalar or vector basis, got [{}]'.format(basis))



class UnknownFunctionInterface:
    def __init__(self):
        pass

class ScalarUnknownFunction(SymbolicFunctionBase, UnknownFunctionInterface):
    def __init__(self, basis, name):
        assert(isinstance(basis, ScalarBasisBase))
        super().__init__(basis, name)


class VectorUnknownFunction(SymbolicVectorFunctionBase, UnknownFunctionInterface):
    def __init__(self, basis, name):
        assert(isinstance(basis, VectorBasisBase))
        super().__init__(basis, name)



def UnknownFunction(basis, name):

    if isinstance(basis, VectorBasisBase):
        return VectorUnknownFunction(basis, name)
    elif isinstance(basis, ScalarBasisBase):
        return ScalarUnknownFunction(basis, name)
    else:
        raise('UnknownFunction expected scalar or vector basis, got [{}]'.format(basis))





if __name__=='__main__':

    v = TestFunction(ScalarBasisBase(), 'v')
    V = TestFunction(VectorBasisBase(VectorShape(2)), 'V')

    print('Function {} has basis {}'.format(v, v.basis()))
    print('Function {} has basis {}'.format(V, V.basis()))
