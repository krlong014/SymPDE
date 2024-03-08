#############################################################################
#
# Base class for constant expressions
#
#############################################################################

from . Expr import Expr
from . ExprShape import ExprShape, ScalarShape, VectorShape
from . IndexableExpr import (
        IndexableExprIterator,
        IndexableExprInterface,
        IndexableExprElementInterface
    )
from numpy.linalg import norm
from numpy import inf, dot, array_equiv
from abc import ABC, abstractmethod
from . ExprEval import ExprEvaluator

class ConstantExprBase(Expr, ABC):
    def __init__(self, data, shape):
        super().__init__(shape)
        self._data = data

    def isConstant(self):
        return True

    def isSpatialConstant(self):
        return True

    def data(self):
        return self._data

    def plusConstant(self, other, sign):
        assert(isinstance(other, ConstantExprBase))
        assert(type(self)==type(other))

        return type(self)(self._data + sign*other._data)

    @abstractmethod
    def timesConstant(self, other):
        pass

    def __str__(self):
        return '{}'.format(self._data)


    def __repr__(self):
        return '{}[data={}, shape={}]'.format(self.typename(),
            self._data, self.shape())

    def _sameas(self, other):
        return array_equiv(self._data, other._data)

    def __eq__(self, other):
        return self.sameas(other)

    def _isZero(self):
        if isinstance(self.shape(), ScalarShape):
            return self._data == 0.0
        return norm(self._data, ord=inf)==0.0


class ConstantScalarExpr(ConstantExprBase):
    def __init__(self, data):
        super().__init__(data, ScalarShape())

    def typename(self):
        return "ConstantScalar"

    def _lessThan(self, other):
        return self.data() < other.data()

    def timesConstant(self, other):
        assert(isinstance(other, ConstantExprBase))

        # We can deal with scalar*scalar here
        if isinstance(other, ConstantScalarExpr):
            return ConstantScalarExpr(self.data() * other.data())

        # Scalar multiplication is always commutative, so we reverse the order
        # to delegate the operation to the other type.
        return other.timesConstant(self)

    def _isZero(self):
        return self.data()==0.0

    def _makeEval(self, context):
        return ConstantScalarEvaluator(self,context)

class ConstantScalarEvaluator(ExprEvaluator):
    def __init__(self,data,context):
        super().__init__(data,context)

    def __str__(self):
        return 'ConstantScalarEvaluator({})'.format(self.myExpr().name())

    def buildAForOrder(self,d):
        Avar = {}
        if d == 0:
            Aconst = {'Identity':1}
        else:
            Aconst = {}
            
        return Aconst, Avar




class ConstantVectorExpr(ConstantExprBase, IndexableExprInterface):
    def __init__(self, data):
        shape = VectorShape(len(data))
        super().__init__(data, shape)


    def typename(self):
        return "ConstantVector"

    def timesConstant(self, other):
        assert(isinstance(other, ConstantExprBase))

        resultShape = ExprShape.productShape(self.shape(), other.shape())

        if isinstance(resultShape, ScalarShape):
            return ConstantScalarExpr(self.data() * other.data())
        if isinstance(resultShape, VectorShape):
            return ConstantVectorExpr(dot(self.data(), other.data()))

    def __getitem__(self, i):
        assert(i>=0 and i<self.shape().dim())
        return self.data()[i]

    def __len__(self):
        return self.shape().dim()

    def __iter__(self):
        return ConstantVectorIterator(self)

    def _lessThan(self, other):
        if len(self) < len(other):
            return True
        if len(self) > len(other):
            return False
        for (mine, yours) in zip(self.data(), other.data()):
            if mine<yours:
                return True
            if yours<mine:
                return False
        return False



class ConstantVectorIterator(IndexableExprIterator):
    '''Iterator for constant vectors'''
    def __init__(self, parent):
        '''Constructor'''
        super().__init__(ConstantVectorExpr, parent)



class ConstantVectorElement(IndexableExprElementInterface):
    '''Element of constant vector.'''
    def __init__(self, parent, index):
        '''Constructor'''
        super().__init__(ConstantVectorExpr, parent, index)
