from SymPDE.Expr import Expr
from SymPDE.ExprShape import ScalarShape, VectorShape
from SymPDE.Coordinate import Coordinate
from abc import ABC, abstractmethod
from numpy.linalg import norm
import itertools as it
from scipy.special import binom
from collections.abc import Iterable
# from SymPDE.DerivSpecifier import DerivSpecifier
#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################


#PROBLEM: Using DerivSpecifier here causes a circular import
    #because of FWB. This means I can't actually change the
    #A/Q set construction in here to use the DerivSpecifier

#QUESTION: Do I need to move the set construction to a new
    #script, or do I need to do something different?

#SOLUTION: 

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

    #intermediate step in part
    # def flatten(lst):
    #     if type(lst) != list:
    #         return lst 

    #     flatList = []

    #     for item in lst:
    #         if type(item) == list:
    #             for x in flatten(item):
    #                 flatList.append(x)
    #         else:
    #             flatList.append(item)

    #     return flatList

    # #intermediate step in intPart
    # def part(n):
    #     if n == 1:
    #         return [0]

    #     parts = [] 

    #     for i in range(n):
    #         for j in range(len(part(i))):
    #             parts.append(flatten([n - i] + [part(i)[j]]))

    #     return parts

    # #builds an integer partition of n
    # def intPart(n):
    #     return [[n]] + part(n)


    # #builds a particular Q set of order d
    # #appends a list of the multiplicities of each derivative to the end
    # def buildFForOrder(self,d):
    #     n = len(self._children)
    #     if d == 1:
    #         F = [i for i in range(n)]
    #     else:
    #         dummyIter = [(i) for i in range(n)]

    #         F = list(it.combinations_with_replacement(dummyIter,d))

    #     mults = [int(binom(d,i)) for i in range(len(F))]
        

    #     FwithMults = {}
    #     for i in range(len(mults)):
    #         FwithMults[F[i]] = mults[i]

    #     return FwithMults

    # #builds all F sets up to oder d
    # def buildAllFUpToOrder(self,d):
    #     n = len(self._children)
    #     assert(n >= 1)

    #     Fsets = {}
    #     for i in range(d):
    #         Fsets = Fsets | self.buildFForOrder(i+1)

    #     return Fsets

    # #builds all Q sets up to order d
    # def buildQUpToOrder(self,d):
    #     Qconst = {}; Qvar = {}

    #     for i in range(d):
    #         [this_Qconst, this_Qvar] = self.buildQForOrder(i+1)
    #         Qconst = Qconst | this_Qconst
    #         Qvar = Qvar | this_Qvar

    #     return(Qconst, Qvar)

    # #builds A set for order d
    # def buildAForOrder(self,order):
    #     n = len(self._children)
    #     assert(n >= 1)

    #     if order == 1:
    #         [Qconst_order_1, Qvar_order_1] = self.buildQForOrder(1)

    #         Aconst = {}; Avar = {}
    #         for q in Qconst_order_1:
    #             [this_Aconst, this_Avar] = self.child(q).buildAForOrder(1)
    #             Aconst = Aconst | this_Aconst
    #             Avar = Avar | this_Avar

    #         for q in Qvar_order_1:
    #             [this_Aconst, this_Avar] = self.child(q).buildAForOrder(1)
    #             Avar = Avar | this_Aconst | this_Avar

    #     if order == 2:
    #         [Qconst_order_1, Qvar_order_1] = self.buildQForOrder(1)

    #         A2_term_const = {}; A2_term_var = {}
    #         for q in Qconst_order_1:
    #             [this_Aconst, this_Avar] = self.child(q).buildAForOrder(2)
    #             A2_term_const = A2_term_const | this_Aconst
    #             A2_term_var = A2_term_var | this_Avar

    #         for q in Qvar_order_1:
    #             [this_Aconst, this_Avar] = self.child(q).buildAForOrder(2)
    #             A2_term_var = A2_term_var | this_Aconst | this_Avar

    #         [Qconst_order_2, Qvar_order_2] = self.buildQForOrder(2)

    #         A1_term_const = {}; A1_term_var = {}
    #         for q in Qconst_order_2:
    #             [this_Aconst_0, this_Avar_0] = self.child(q[0]).buildAForOrder(1)
    #             [this_Aconst_1, this_Avar_1] = self.child(q[1]).buildAForOrder(1)
    #             this_Aconst = list(zip(this_Aconst_0, this_Aconst_1))
    #             this_Avar = list(zip(this_Avar_0, this_Avar_1))

    #             for item in this_Aconst:
    #                 A1_term_const[item] = Qconst_order_2[item]
    #             for item in this_Avar:
    #                 A1_term_var[item] = Qvar_order_2[item]

    #         for q in Qvar_order_2:
    #             [this_Aconst_0, this_Avar_0] = self.child(q[0]).buildAForOrder(1)
    #             [this_Aconst_1, this_Avar_1] = self.child(q[1]).buildAForOrder(1)
    #             this_Aconst = list(zip(this_Aconst_0, this_Aconst_1))
    #             this_Avar = list(zip(this_Avar_0, this_Avar_1))

    #             for item in this_Aconst:
    #                 A1_term_var[item] = Qvar_order_2[item]
    #             for item in this_Avar:
    #                 A1_term_var[item] = Qvar_order_2[item]

    #         Aconst = A1_term_const | A2_term_const
    #         Avar = A1_term_var | A2_term_var 
    #     if order == 3:
    #         [Qconst_order_1, Qvar_order_1] = self.buildQForOrder(1)
    #         Q1 = Qconst_order_1 | Qvar_order_1

    #         A3_term_const = {}; A3_term_var = {}
    #         for q in Qconst_order_1:
    #             [this_Aconst,this_Avar] = self.child(q).buildAForOrder(3)
    #             A3_term_const = A3_term_const | this_Aconst
    #             A3_term_var = A3_term_var | this_Avar

    #         [Qconst_order_2, Qvar_order_2] = self.buildQForOrder(2)
    #         Q2 = Qconst_order_2 | Qvar_order_2

    #         A2_term_const = {}; A2_term_var = {}
    #         for q in Qconst_order_2:
    #             [this_Aconst_0,this_Avar_0] = self.child(q[0]).buildAForOrder(1)
    #             [this_Aconst_1,this_Avar_1] = self.child(q[1]).buildAForOrder(2)
    #             this_Aconst = list(zip(this_Aconst_0, this_Aconst_1))
    #             this_Avar = list(zip(this_Avar_0,this_Avar_1))
    #             for item in this_Aconst:
    #                 A2_term_const[item] = Q2[item]
    #             for item in this_Avar:
    #                 A2_term_var[item] = Q2[item]

    #         [Qconst_order_3, Qvar_order_3] = self.buildQForOrder(3)
    #         Q3 = Qconst_order_3 | Qvar_order_3

    #         A1_term_const = {}; A1_term_var = {}
    #         for q in Qconst_order_3:
    #             [this_Aconst_0,this_Avar_0] = self.child(q[0]).buildAForOrder(1)
    #             [this_Aconst_1,this_Avar_1] = self.child(q[1]).buildAForOrder(1)
    #             [this_Aconst_2,this_Avar_2] = self.child(q[2]).buildAForOrder(1)
    #             this_Aconst = list(zip(this_Aconst_0,this_Aconst_1,this_Aconst_2))
    #             this_Avar = list(zip(this_Avar_0,this_Avar_1,this_Avar_2))
    #             for item in this_Aconst:
    #                 A1_term_const[item] = Q3[item]
    #             for item in this_Avar:
    #                 A1_term_var[item] = Q3[item]

    #         A1_term_var = Q3
    #         for q in Q3:
    #             if q in A1_term_const:
    #                 del A1_term_var[q]

    #         Aconst = A3_term_const | A2_term_const | A1_term_const

    #         Avar = A3_term_var | A2_term_var | A1_term_var 
    #     return Aconst, Avar 

    # #builds all A sets up to order d
    # def buildAUpToOrder(self,order):
    #     Aconst = {}; Avar = {}
    #     for i in range(order):
    #         [this_Aconst, this_Avar] = self.buildAForOrder(i+1)
    #         Aconst = Aconst | this_Aconst
    #         Avar = Avar | this_Avar
    #     return Aconst, Avar

    # #builds a dictionary of required derivatives with order d and 
    # #their multiplicites given petitioning sets
    # def buildR(self,Petitions):
    #     max_order = len(Petitions)

    #     Rconst = []; Rvar = []
    #     for d in range(1,max_order+1):
    #         [this_Aconst, this_Avar] = self.buildAForOrder(d)
    #         RC = {}; RV = {}
    #         for P in Petitions[d-1]:
    #             for item in this_Aconst:
    #                 if (item == P) or (isinstance(P,Iterable) and item in P):
    #                     RC[item] = this_Aconst[item]
    #             for item in this_Avar:
    #                 if (item == P) or (isinstance(P,Iterable) and item in P):
    #                     RV[item] = this_Avar[item]

    #         if RC != {}:
    #             Rconst.append(RC)
            
    #         if RV != {}:
    #             Rvar.append(RV)

    #     return Rconst, Rvar

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
        self.arg = arg 

    def __str__(self):
        return '-%s' % Expr._maybeParenthesize(self.arg())

    def __repr__(self):
        return 'UnaryMinus[arg={}]'.format(self.arg().__repr__())

    def isLinearInTests(self):
        return self.arg().isLinearInTests()

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
