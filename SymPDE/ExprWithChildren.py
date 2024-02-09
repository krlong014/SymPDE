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

    def makeEvalsForChildren(self, context):
        
        childEvals = []
        for i,c in enumerate(self.children()):
            childEvals.append(c.makeEval(context))
        
        return childEvals
            
            

class UnaryExpr(ExprWithChildren):
    def __init__(self, arg, shape):
        super().__init__((arg,),shape)

    def arg(self):
        return self.child(0)

    def _sameas(self, other):
        return self.arg().sameas(other.arg())
    
    def makeArgEval(self, context):
        return self.arg().makeEval(context)


class BinaryExpr(ExprWithChildren):
    def __init__(self, L, R, shape):
        super().__init__((L, R), shape)

    def left(self):
        return self.child(0)

    def right(self):
        return self.child(1)

    def isSpatialConstant(self):
        return self.left().isSpatialConstant() and self.right().isSpatialConstant()
    
    def makeLeftEval(self, context):
        return self.left().makeEval(context)
    
    def makeRightEval(self, context):
        return self.right().makeEval(context)
    
