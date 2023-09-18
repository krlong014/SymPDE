from . Expr import Expr
from . ExprShape import ScalarShape, VectorShape
from abc import ABC, abstractmethod
from numpy.linalg import norm

#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################

class ExprWithChildren(Expr):
    '''ExprWithChildren is a base class for operations that act on expressions.'''
    def __init__(self, children, shape):
        '''Constructor for ExprWithChildren.'''
        if not (isinstance(children, (list, tuple)) and len(children)>0):
            raise ValueError('ExprWithChildren bad input {}'.format(children))
        for c in children:
            assert(not c.isAggregate())

        super().__init__(shape)

        self._children = children

    def children(self):
        return self._children

    def child(self, i):
        return self._children[i]

    def _sameas(self, other):
        for me, you in zip(self._children, other._children):
            if not me._sameas(you):
                return False
        return True

    def _lessThan(self, other):
        if len(self._children) < len(other._children):
            return True
        if len(self._children) > len(other._children):
            return False

        for mine, yours in zip(self._children, other._children):
            if mine.lessThan(yours):
                return True
            if yours.lessThan(mine):
                return False
        return False

    def isSpatialConstant(self):
        for c in self._children:
            if not c.isSpatialConstant():
                return False
        return True

    def isConstant(self):
        for c in self._children:
            if not c.isConstant():
                return False
        return True

    def hasTest(self):
        for c in self._children:
            if c.hasTest():
                return True
        return False

    def hasUnknown(self):
        for c in self._children:
            if c.hasUnknown():
                return True
        return False

    def isIndependentOf(self, u):
        for c in self._children:
            if not c.isIndependentOf(u):
                return False
        return True

    def getTests(self):
        rtn = set()
        for c in self._children:
            s = c.getTests()
            rtn = rtn.union(s)
        return rtn

    def getUnks(self):
        rtn = set()
        for c in self._children:
            s = c.getUnks()
            rtn = rtn.union(s)
        return rtn


class UnaryExpr(ExprWithChildren):
    def __init__(self, arg, shape):
        super().__init__((arg,),shape)

    def arg(self):
        return self.child(0)

    def _sameas(self, other):
        return self.arg().sameas(other.arg())



class UnaryMinus(UnaryExpr):
    def __init__(self, arg):
        super().__init__(arg, Expr._getShape(arg))

    def __str__(self):
        return '-%s' % Expr._maybeParenthesize(self.arg())

    def __repr__(self):
        return 'UnaryMinus[arg={}]'.format(self.arg().__repr__())

    def isLinearInTests(self):
        return self.arg().isLinearInTests()



class BinaryExpr(ExprWithChildren):
    def __init__(self, L, R, shape):
        super().__init__((L, R), shape)

    def left(self):
        return self.child(0)

    def right(self):
        return self.child(1)

    def isSpatialConstant(self):
        return self.left().isSpatialConstant() and self.right().isSpatialConstant()

class BinaryArithmeticOp(BinaryExpr):
    def __init__(self, L, R, shape):
        super().__init__(L, R, shape)

    @abstractmethod
    def opString(self):
        pass

    def __str__(self):
        return '{}{}{}'.format(Expr._maybeParenthesize(self.left()),
            self.opString(), Expr._maybeParenthesize(self.right()))


class SumExpr(BinaryArithmeticOp):
    def __init__(self, L, R, sign):
        super().__init__(L, R, Expr._getShape(L))
        self.sign = sign

    def opString(self):
        if self.sign > 0:
            return '+'
        else:
            return '-'

    def _sameas(self, other):
        return (self.left().sameas(other.left()) and self.right().sameas(other.right())
            and self.sign==other.sign)

    def _lessThan(self, other):
        if self.sign < other.sign:
            return True
        if self.sign > other.sign:
            return False
        return super()._lessThan(other)

    def __repr__(self):
        return 'SumExpr[left={}, right={}, shape={}]'.format(self.left().__repr__(),
            self.right().__repr__(), self.shape())

    def everyTermHasTest(self):
        return self.left().hasTest() and self.right().hasTest()

    def isLinearInTests(self):
        return (self.left().isLinearInTests()
                and self.right().isLinearInTests())


class ProductExpr(BinaryArithmeticOp):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())

    def opString(self):
        return '*'

    def __repr__(self):
        return 'ProductExpr[left={}, right={}]'.format(self.left().__repr__(),
            self.right().__repr__())

    def isLinearInTests(self):
        if ( (self.left().isLinearInTests() and not self.right().hasTest())
            or
            (self.right().isLinearInTests() and not self.left().hasTest()) ):
            return True
        return False



def Dot(a, b):
    assert(isinstance(a.shape(), VectorShape)
        and isinstance(b.shape(), VectorShape))
    assert(a.shape().dim() == b.shape().dim())

    return DotProductExpr(a,b)


class DotProductExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ExprShape.productShape(L.shape(), R.shape()))

    def opString(self):
        return 'dot'

    def __str__(self):
        return 'dot({},{})'.format(self.left(), self.right())

    def __repr__(self):
        return 'DotProductExpr[left={}, right={}, shape={}]'.format(self.left().__repr__(),
            self.right().__repr__(), self.shape())

    def isLinearInTests(self):
        if ( (left.isLinearInTests() and not right.hasTests())
            or
            (right.isLinearInTests() and not left.hasTests()) ):
            return True
        return False

def Cross(a, b):
    assert(isinstance(a.shape(), VectorShape)
        and isinstance(b.shape(), VectorShape))
    assert(a.shape().dim() == b.shape().dim())

    return CrossProductExpr(a,b,a.shape())

class CrossProductExpr(BinaryExpr):
    def __init__(self, L, R, shape):
        pass

    def __str__(self):
        return 'cross({},{})'.format(self.L, self.R)

    def isLinearInTests(self):
        if ( (left.isLinearInTests() and not right.hasTests())
            or
            (right.isLinearInTests() and not left.hasTests()) ):
            return True
        return False

class QuotientExpr(BinaryArithmeticOp):
    def __init__(self, L, R):
        super().__init__(L, R, L.shape())

    def opString(self):
        return '/'

    def isLinearInTests(self):
        if left.isLinearInTests() and not right.hasTests():
            return True
        return False



class PowerExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())

    def __str__(self):
        return 'pow({},{})'.format(self.left(), self.right())
