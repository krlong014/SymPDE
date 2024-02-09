from SymPDE.Expr import Expr
from SymPDE.ExprWithChildren import ExprWithChildren, UnaryExpr, BinaryExpr
from SymPDE.ExprShape import ScalarShape, VectorShape
from abc import abstractmethod
from numpy.linalg import norm
import itertools as it
from scipy.special import binom
from collections.abc import Iterable

#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################




class UnaryMinus(UnaryExpr):
    def __init__(self, arg):
        super().__init__(arg, Expr._getShape(arg))
        self.arg = arg 

    def __str__(self):
        return '-%s' % Expr._maybeParenthesize(self.arg())

    def __repr__(self):
        return 'UnaryMinus[arg={}]'.format(self.arg().__repr__())

    def isLinearInTests(self):
        return self.arg().isLinearInTests()
    
    def _makeEval(self, context):
        return 

    # def buildQForOrder(self,d):
    #     F = self.buildFForOrder(d)
        
    #     Qvar = {}
    #     if d == 1:
    #         Qconst = {self.arg:1}
    #     else:
    #         Qconst = {}

    #     return Qconst, Qvar


    # def buildA(self,d):
    #     [Aconst, Avar] = self.arg.buildA()
    #     return Aconst, Avar

    # def printQ(self,d):
    #     [Qconst, Qvar] = self.buildQ(d)

    #     for Q in Qconst:
    #         print("Qconst contains the derivative {}, which has multiplicity {}".format(Q,Qconst[Q]))
    #     for Q in Qvar:
    #         print("Qvar contains the derivative {}, which has multiplicity {}".format(Q,Qvar[Q]))


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
        self.L = L 
        self.R = R 
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

    def buildF(self,d):
        Fsets = self.buildAllFUpToOrder(d)
        return Fsets

    def buildQForOrder(self,d):
        F = self.buildFForOrder(d)
        F_keys = F.keys()

        Qconst = {}; Qvar = {}
        const_keys_to_add = []
        if d == 1:
            for key in F_keys:
                const_keys_to_add.append(key)
        
        for key in const_keys_to_add:
            Qconst[key] = F[key]

        return Qconst, Qvar 

    def printQ(self,d):
        [Qconst, Qvar] = self.buildQ(d)

        for Q in Qconst:
            print("Qconst contains the derivative {}, which has multiplicity {}".format(Q,Qconst[Q]))
        for Q in Qvar:
            print("Qvar contains the derivative {}, which has multiplicity {}".format(Q,Qvar[Q]))



class ProductExpr(BinaryArithmeticOp):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())
        self.L = L 
        self.R = R 

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

    # def buildQForOrder(self,d):
    #     F = self.buildFForOrder(d)
    #     F_keys = F.keys()

    #     Qconst = {}; Qvar = {}
    #     if d == 1:
    #         Qvar = F 
    #     elif d == 2:
    #         const_keys_to_add = []
    #         for key in F_keys:
    #             if key[0] != key[1]:
    #                 const_keys_to_add.append(key)

    #         for key in const_keys_to_add:
    #             Qconst[key] = F[key]

    #     return Qconst, Qvar 

    # def printQ(self,d):
    #     [Qconst, Qvar] = self.buildQ(d)

    #     for Q in Qconst:
    #         print("Qconst contains the derivative {}, which has multiplicity {}".format(Q,Qconst[Q]))
    #     for Q in Qvar:
    #         print("Qvar contains the derivative {}, which has multiplicity {}".format(Q,Qvar[Q]))


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
        self.L = L 
        self.R = R

    def opString(self):
        return '/'

    def isLinearInTests(self):
        if left.isLinearInTests() and not right.hasTests():
            return True
        return False

    # def buildQForOrder(self,d):
    #     F = self.buildFForOrder(d)
    #     F_keys = F.keys()

    #     Qconst = {}; Qvar = F 
    #     if d >= 2:
    #         keys_to_remove = []
    #         for Q in Qvar:
    #             num_zeros = Q.count(0)
    #             if num_zeros >= 2:
    #                 keys_to_remove.append(Q)

    #         for key in keys_to_remove:
    #             del Qvar[key]

    #     return Qconst, Qvar 

    # def printQ(self,d):
    #     [Qconst, Qvar] = self.buildQ(d)

    #     for Q in Qconst:
    #         print("Qconst contains the derivative {}, which has multiplicity {}".format(Q,Qconst[Q]))
    #     for Q in Qvar:
    #         print("Qvar contains the derivative {}, which has multiplicity {}".format(Q,Qvar[Q]))

class PowerExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())
        self.L = L 
        self.R = R

    def __str__(self):
        return 'pow({},{})'.format(self.left(), self.right())

    # def buildQForOrder(self,d):
    #     F = self.buildFForOrder(d)

    #     Qconst = {}; Qvar = F 
    #     return Qconst, Qvar 

    # def printQ(self,d):
    #     [Qconst, Qvar] = self.buildQ(d)

    #     for Q in Qconst:
    #         print("Qconst contains the derivative {}, which has multiplicity {}".format(Q,Qconst[Q]))
    #     for Q in Qvar:
    #         print("Qvar contains the derivative {}, which has multiplicity {}".format(Q,Qvar[Q]))
