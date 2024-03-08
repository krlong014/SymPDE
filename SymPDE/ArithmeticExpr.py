from SymPDE.Expr import Expr
from SymPDE.ExprWithChildren import UnaryExpr, BinaryExpr
from SymPDE.ExprShape import ExprShape, ScalarShape, VectorShape
from abc import abstractmethod
import itertools as it
from collections.abc import Iterable

#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################


class UnaryMinus(UnaryExpr):
    def __init__(self, arg):
        super().__init__(arg, Expr._getShape(arg))
        
    def __str__(self):
        myArg = self.arg()
        return '-%s' % Expr._maybeParenthesize(myArg)

    def __repr__(self):
        return 'UnaryMinus[arg={}]'.format(self.arg().__repr__())

    def isLinearInTests(self):
        return self.arg().isLinearInTests()
    
    def _makeEval(self, context):
        from SymPDE.ChainRuleEval import UnaryMinusEvaluator
        return UnaryMinusEvaluator(self, context)

    

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
    
    def _makeEval(self, context):
        from SymPDE.ChainRuleEval import SumEvaluator
        return SumEvaluator(self, context)



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

    def _makeEval(self,context):
        from SymPDE.ChainRuleEval import ProductEvaluator
        return ProductEvaluator(self,context)

    
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
        if ( (self.left().isLinearInTests() and not self.right().hasTests())
            or
            (self.right().isLinearInTests() and not self.left().hasTests()) ):
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
        if ( (self.left().isLinearInTests() and not self.right().hasTests())
            or
            (self.right().isLinearInTests() and not self.left().hasTests()) ):
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
        if self.left().isLinearInTests() and not self.right().hasTests():
            return True
        return False
    
    def _makeEval(self,context):
        from SymPDE.ChainRuleEval import QuotientEvaluator
        return QuotientEvaluator(self,context)

class PowerExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())
        self.L = L 
        self.R = R

    def __str__(self):
        return 'pow({},{})'.format(self.left(), self.right())

    
    def _makeEval(self,context):
        from SymPDE.ChainRuleEval import PowerEvaluator
        return PowerEvaluator(self,context)
