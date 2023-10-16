from . Expr import Expr
from . ExprShape import ScalarShape, VectorShape
from SymPDE.Coordinate import Coordinate
from abc import ABC, abstractmethod
from numpy.linalg import norm
import itertools as it
from scipy.special import binom

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

    #intermediate step in part
    def flatten(lst):
        if type(lst) != list:
            return lst 

        flatList = []

        for item in lst:
            if type(item) == list:
                for x in flatten(item):
                    flatList.append(x)
            else:
                flatList.append(item)

        return flatList

    #intermediate step in intPart
    def part(n):
        if n == 1:
            return [0]

        parts = [] 

        for i in range(n):
            for j in range(len(part(i))):
                parts.append(flatten([n - i] + [part(i)[j]]))

        return parts

    #builds an integer partition of n
    def intPart(n):
        return [[n]] + part(n)


    #builds a particular Q set of order d
    #appends a list of the multiplicities of each derivative to the end
    def buildFForOrder(self,d):
        n = len(self._children)
        if d == 1:
            F = [i for i in range(n)]
        else:
            dummyIter = [(i) for i in range(n)]

            F = list(it.combinations_with_replacement(dummyIter,d))

        mults = [int(binom(d,i)) for i in range(len(F))]
        

        FwithMults = {}
        for i in range(len(mults)):
            FwithMults[F[i]] = mults[i]

        return FwithMults

    #builds all Q sets of order d or less
    def buildAllFUpToOrder(self,d):
        n = len(self._children)
        assert(n >= 1)

        Fsets = {}
        for i in range(d):
            Fsets = Fsets | self.buildFForOrder(i+1)

        return Fsets

    #builds a particular A set of order d
    def buildAForOrder(self,d):
        n = len(self._children)
        assert(n >= 1)

        A = []
        for i in range(n):
            A.append(self.child(i).buildAForOrder(d))

        return A

    #builds all A sets of order d or less
    def buildAllAUpToOrder(self,d):
        n = len(self._children)
        assert (n >= 1)

        Asets = []
        for i in range(d):
            Asets.append(self.buildAForOrder(i+1))

        return Asets


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

    def buildQ(self,d):
        Fsets = self.buildAllFUpToOrder(d)
        Fsets_keys = list(Fsets.keys())

        f_key = Fsets_keys[0]

        Qconst = {}
        Qconst[f_key] = Fsets[f_key]
        Qvar = {}

        return Qconst,Qvar

    def buildA(self,d):
        [Aconst, Avar] = self.arg.buildA()
        return Aconst, Avar



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

    def buildQ(self,d):
        Fsets = self.buildAllFUpToOrder(d)
        Fsets_keys = Fsets.keys()
        # Qconst = Fsets[0]

        Qconst = {}; Qvar = {}
        const_keys_to_add = []
        for key in Fsets_keys:
            if type(key) == int:
                const_keys_to_add.append(key)

        for key in const_keys_to_add:
            Qconst[key] = Fsets[key]

        return [Qconst,Qvar]

    def buildA(self):
        [LeftAconst, LeftAvar] = self.L.buildA()
        [RightAconst, RightAvar] = self.R.buildA()

        A = LeftAconst | LeftAvar | RightAvar | RightAconst
        a_keys = A.keys()

        Aconst = {}
        a_const_keys_to_add = LeftAconst.keys() & RightAconst.keys()
        for a in a_const_keys_to_add:
            Aconst[a] = A[a]
        
        Avar = {}
        a_var_keys_to_add = []
        for a in a_keys:
            if a not in Aconst:
                a_var_keys_to_add.append(a)

        for a in a_var_keys_to_add:
            Avar[a] = A[a]

        return Aconst, Avar

    # def buildR(self,d):
        


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

    def buildQ(self, d):
        assert(type(d)==int)
        Fsets = self.buildAllFUpToOrder(d)
        Fsets_keys = Fsets.keys()

        Qconst = {}; Qvar = {}
        if d == 1:
            Qvar = Fsets 
        
        const_keys_to_add = []; var_keys_to_add = []
        for f_key in Fsets_keys:
            if type(f_key) == int:
                var_keys_to_add.append(f_key)
            elif len(f_key) == 2:
                if f_key[0] != f_key[1]:
                    const_keys_to_add.append(f_key)

        for c_key in const_keys_to_add:
            Qconst[c_key] = Fsets[c_key]

        for v_key in var_keys_to_add:
            Qvar[v_key] = Fsets[v_key]

        return Qconst,Qvar

    def buildA(self):
        [LeftAconst, LeftAvar] = self.L.buildA()
        [RightAconst, RightAvar] = self.R.buildA()


        A = LeftAconst | LeftAvar | RightAconst | RightAvar
        # print("A = ",A)
        a_keys = A.keys()

        a_const_keys_to_add = LeftAconst.keys() & RightAconst.keys()
        Aconst = {}
        for a in a_const_keys_to_add:
            Aconst[a] = A[a]

        a_var_keys_to_add = []
        Avar = {}
        for a in a_keys:
            if a not in Aconst:
                a_var_keys_to_add.append(a)

        for a in a_var_keys_to_add:
            Avar[a] = A[a]

        return Aconst, Avar 


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

    def buildQ(self, d):
        Fsets = self.buildAllFUpToOrder(d)
        
        #none are constant
        Qconst = {}

        Qvar = Fsets

        Qvar_keys = Qvar.keys()
        keys_to_remove = []

        for Q in Qvar_keys:
            if type(Q) == int:
                continue
            else:
                num_zeros = Q.count(0)
                if num_zeros >= 2:
                    keys_to_remove.append(Q)

        for key in keys_to_remove:
            del Qvar[key]

        return([Qconst,Qvar])

    def buildA(self):
        [LeftAconst, LeftAvar] = self.L.buildA()
        [RightAconst, RightAvar] = self.R.buildA()

        A = LeftAconst | LeftAvar | RightAconst | RightAvar
        a_keys = A.keys()

        a_const_keys_to_add = LeftAconst.keys() & RightAconst.keys()
        Aconst = {}
        for a in a_const_keys_to_add:
            Aconst[a] = A[a]

        a_var_keys_to_add = []
        Avar = {}
        for a in a_keys:
            if a not in Aconst:
                a_var_keys_to_add.append(a)

        for a in a_var_keys_to_add:
            Avar[a] = A[a]

        return Aconst, Avar 


class PowerExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())
        self.L = L 
        self.R = R

    def __str__(self):
        return 'pow({},{})'.format(self.left(), self.right())

    def buildQ(self, d):
        Fsets = self.buildAllFUpToOrder(d)
        #no derivatives are constant
        Qconst = {}
        #all derivatives are variable
        Qvar = Fsets

        return([Qconst,Qvar])

    def buildA(self):
        [LeftAconst, LeftAvar] = self.L.buildA()
        [RightAconst, RightAvar] = self.R.buildA()

        A = LeftAconst | LeftAvar | RightAconst | RightAvar
        a_keys = A.keys()

        a_const_keys_to_add = LeftAconst.keys() & RightAconst.keys()
        Aconst = {}
        for a in a_const_keys_to_add:
            Aconst[a] = A[a]

        a_var_keys_to_add = []
        Avar = {}
        for a in a_keys:
            if a not in Aconst:
                a_var_keys_to_add.append(a)

        for a in a_var_keys_to_add:
            Avar[a] = A[a]

        return Aconst, Avar 
