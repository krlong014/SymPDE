from . Expr import Expr
from . IndexableExpr import (
        IndexableExprIterator,
        IndexableExprInterface,
        IndexableExprElementInterface
    )
from . ExprShape import (ExprShape, ScalarShape, TensorShape,
        VectorShape, AggShape)
from . BasisBase import BasisBase, VectorBasisBase, ScalarBasisBase
from . DiscreteSpaceBase import DiscreteSpaceBase

import pytest

class FunctionWithBasis(Expr):

    _nextFuncID = 0

    def __init__(self, basis, name):
        super().__init__(basis.shape())
        self._name = name
        self._funcID = FunctionWithBasis._nextFuncID
        FunctionWithBasis._nextFuncID += 1
        self._basis = basis

    def basis(self):
        return self._basis

    def _sameas(self, other):
        return self._funcID == other._funcID

    def __str__(self):
        return self._name

    def isTest(self):
        return False

    def isUnknown(self):
        return False

    def isDiscrete(self):
        return False

    def funcID(self):
        return self._funcID

    def name(self):
        return self._name

    def _lessThan(self, other):
        return self._funcID < other._funcID

    def hasTest(self):
        return self.isTest()

    def hasUnknown(self):
        return self.isUnknown()

    def isLinearInTests(self):
        return self.isTest()

    def getTests(self):
        if self.isTest():
            return {self}
        return set()

    def getUnks(self):
        if self.isUnknown():
            return {self}
        return set()



class FunctionWithScalarBasis(FunctionWithBasis):

    def __init__(self, basis, name):

        assert(isinstance(basis.shape(), ScalarShape))
        super().__init__(basis, name)

class FunctionWithVectorBasis(FunctionWithBasis, IndexableExprInterface):

    def __init__(self, basis, name, myType):

        assert(isinstance(basis.shape(), VectorShape))

        super().__init__(basis, name)

        self._elems = []
        for i in range(basis.shape().dim()):
            if myType=='Test':
                self._elems.append(TestFunctionElement(self, i))
            elif myType=='Unknown':
                self._elems.append(UnknownFunctionElement(self, i))
            elif myType=='Discrete':
                self._elems.append(DiscreteFunctionElement(self, i))
            else:
                raise ValueError(
                    'Invalid type {} in FunctionWithVectorBasis'.format(myType)
                    )

    def __getitem__(self, i):
        return self._elems[i]

    def __iter__(self):
        return VectorFunctionIterator(self)


class VectorFunctionIterator(IndexableExprIterator):
    '''Iterator for elements of vector-valued function.'''
    def __init__(self, subtype, parent):
        '''Constructor'''
        super().__init__(subtype, parent)


class VectorFunctionElement(Expr, IndexableExprElementInterface):

    def __init__(self, subtype, parent, myIndex):
        super().__init__(ScalarShape())
        super(Expr, self).__init__(subtype, parent, myIndex)


    def _sameas(self, other):
        return (self.parent().sameas(other.parent()) and
            self.index())

    def _lessThan(self, other):
        if self.parent().lessThan(other.parent()):
            return True
        if other.parent().lessThan(self.parent()):
            return False
        return self.index() < other.index()

    def isTest(self):
        return False

    def isUnknown(self):
        return False

    def isDiscrete(self):
        return False



# -----------------------------------------------------------------------------
# Test function
# -----------------------------------------------------------------------------

class ScalarTestFunction(FunctionWithScalarBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name)

    def isTest(self):
        return True

    def __repr__(self):
        return 'ScalarTestFunction(name={}, bas={})'.format(self._name,
                                                            self.basis())
    def buildA(self,d):
        Aconst = {self._name : 1}
        Avar = {}
        return Aconst, Avar 





class VectorTestFunction(FunctionWithVectorBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name, 'Test')

    def isTest(self):
        return True

    def __repr__(self):
        return 'VectorTestFunction(name={}, bas={})'.format(self.name(),
                                                            self.basis())


class TestFunctionElement(VectorFunctionElement):
    # Need to keep pytest from thinking this is a class of unit tests
    __test__=False
    def __init__(self, parent, myIndex):
        super().__init__(VectorTestFunction, parent, myIndex)

    def isTest(self):
        return True

def TestFunction(basis, name):

    if isinstance(basis, VectorBasisBase):
        return VectorTestFunction(basis, name)
    elif isinstance(basis, ScalarBasisBase):
        return ScalarTestFunction(basis, name)
    else:
        raise('TestFunction expected scalar or vector basis, got [{}]'.format(basis))



# -----------------------------------------------------------------------------
# Unknown function
# -----------------------------------------------------------------------------


class ScalarUnknownFunction(FunctionWithScalarBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name)

    def isUnknown(self):
        return True

    def __repr__(self):
        return 'ScalarUnknownFunction(name={}, bas={})'.format(self.name(),
                                                            self.basis())

    def buildA(self,d):
        Aconst = {self._name : 1}
        Avar = {}
        return Aconst, Avar 

class VectorUnknownFunction(FunctionWithVectorBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name, 'Unknown')

    def isUnknown(self):
        return True

    def __repr__(self):
        return 'VectorUnknownFunction(name={}, bas={})'.format(self.name(),
                                                            self.basis())


class UnknownFunctionElement(VectorFunctionElement):
    def __init__(self, parent, myIndex):
        super().__init__(VectorUnknownFunction, parent, myIndex)

    def isUnknown(self):
        return True

def UnknownFunction(basis, name):

    if isinstance(basis, VectorBasisBase):
        return VectorUnknownFunction(basis, name)
    elif isinstance(basis, ScalarBasisBase):
        return ScalarUnknownFunction(basis, name)
    else:
        raise('UnknownFunction expected scalar or vector basis, got [{}]'.format(basis))




# -----------------------------------------------------------------------------
# Discrete function
# -----------------------------------------------------------------------------


class ScalarDiscreteFunction(FunctionWithScalarBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name)
        self._vector = None

    def setVector(self, vec):
        self._vector = vec

    def getVector(self):
        return self._vector

    def isDiscrete(self):
        return True


class VectorDiscreteFunction(FunctionWithVectorBasis):
    def __init__(self, basis, name):
        assert(isinstance(basis, VectorBasisBase))
        super().__init__(basis, name, 'Discrete')

    def isDiscrete(self):
        return True


class DiscreteFunctionElement(VectorFunctionElement):

    def __init__(self, parent, myIndex):
        super().__init__(VectorDiscreteFunction, parent, myIndex)

    def isDiscrete(self):
        return True


def DiscreteFunction(space, name):

    basis = space.basis()
    if isinstance(basis, VectorBasisBase):
        return VectorDiscreteFunction(basis, name)
    elif isinstance(basis, ScalarBasisBase):
        return ScalarDiscreteFunction(basis, name)
    else:
        raise('UnknownFunction expected scalar or vector basis, got [{}]'.format(basis))


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------



if __name__=='__main__':

    v = TestFunction(ScalarBasisBase(0), 'v')
    V = TestFunction(VectorBasisBase(0,VectorShape(2)), 'V')

    print('Function {} has basis {}'.format(v, v.basis()))
    print('Function {} has basis {}'.format(V, V.basis()))
    print('V0.isTest()=', V.isTest())
    print('V0.isUnknown()=', V.isUnknown())

    v0 = V[0]
    v1 = V[1]
    print('v0.isTest()=', v0.isTest())
    print('v1.isTest()=', v1.isTest())
