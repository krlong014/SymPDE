from Expr import Expr, VectorExprInterface, VectorElementInterface
from ExprShape import (ExprShape, ScalarShape, TensorShape,
        VectorShape, AggShape)
from BasisBase import BasisBase, VectorBasisBase, ScalarBasisBase
from DiscreteSpaceBase import DiscreteSpaceBase

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

    def _lessThan(self, other):
        return self._funcID < other._funcID



class FunctionWithScalarBasis(FunctionWithBasis):

    def __init__(self, basis, name):

        assert(isinstance(basis.shape(), ScalarShape))
        super().__init__(basis, name)

class FunctionWithVectorBasis(FunctionWithBasis, VectorExprInterface):

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


class VectorFunctionElement(Expr, VectorElementInterface):

    def __init__(self, parent, myIndex):
        super().__init__(ScalarShape())
        super(Expr, self).__init__(parent, myIndex)


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


class VectorTestFunction(FunctionWithVectorBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name, 'Test')

    def isTest(self):
        return True


class TestFunctionElement(VectorFunctionElement):
    # Need to keep pytest from thinking this is a class of unit tests
    __test__=False
    def __init__(self, parent, myIndex):
        super().__init__(parent, myIndex)

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


class VectorUnknownFunction(FunctionWithVectorBasis):
    def __init__(self, basis, name):
        super().__init__(basis, name, 'Unknown')

    def isUnknown(self):
        return True


class UnknownFunctionElement(VectorFunctionElement):
    def __init__(self, parent, myIndex):
        super().__init__(parent, myIndex)

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
        super().__init__(parent, myIndex)

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



class TestFunctionIdentification:

    def test_TestFunc(self):
        v = TestFunction(ScalarBasisBase(0), 'v')
        V = TestFunction(VectorBasisBase(0,VectorShape(2)), 'V')

        v0 = V[0]
        v1 = V[1]

        vIsTest = v.isTest()
        VIsTest = V.isTest()
        v0IsTest = v0.isTest()
        v1IsTest = v1.isTest()

        vIsNotUnknown = not v.isUnknown()
        VIsNotUnknown = not V.isUnknown()
        v0IsNotUnknown = not v0.isUnknown()
        v1IsNotUnknown = not v1.isUnknown()

        v0IndexIsZero = v0.index()==0
        v1IndexIsOne = v1.index()==1

        assert(vIsTest and VIsTest and v0IsTest and v1IsTest
            and vIsNotUnknown and VIsNotUnknown
            and v0IsNotUnknown and v1IsNotUnknown
            and v0IndexIsZero and v1IndexIsOne)

    def test_UnknownFunc(self):
        v = UnknownFunction(ScalarBasisBase(0), 'v')
        V = UnknownFunction(VectorBasisBase(0,VectorShape(2)), 'V')

        v0 = V[0]
        v1 = V[1]

        vIsUnknown = v.isUnknown()
        VIsUnknown = V.isUnknown()
        v0IsUnknown = v0.isUnknown()
        v1IsUnknown = v1.isUnknown()

        vIsNotTest = not v.isTest()
        VIsNotTest = not V.isTest()
        v0IsNotTest = not v0.isTest()
        v1IsNotTest = not v1.isTest()

        v0IndexIsZero = v0.index()==0
        v1IndexIsOne = v1.index()==1

        assert(vIsUnknown and VIsUnknown and v0IsUnknown and v1IsUnknown
            and vIsNotTest and VIsNotTest
            and v0IsNotTest and v1IsNotTest
            and v0IndexIsZero and v1IndexIsOne)

    def test_DiscreteFunc(self):
        v = DiscreteFunction(DiscreteSpaceBase(ScalarBasisBase(0)), 'v')
        V = DiscreteFunction(DiscreteSpaceBase(VectorBasisBase(0,VectorShape(2))), 'V')

        v0 = V[0]
        v1 = V[1]

        vIsDiscrete = v.isDiscrete()
        VIsDiscrete = V.isDiscrete()
        v0IsDiscrete = v0.isDiscrete()
        v1IsDiscrete = v1.isDiscrete()

        assert(vIsDiscrete and VIsDiscrete and v0IsDiscrete and v1IsDiscrete)
