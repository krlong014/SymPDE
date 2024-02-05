from abc import ABC, abstractmethod
from . ExprShape import (ExprShape, ScalarShape, TensorShape,
    VectorShape, AggShape)
from numbers import Number
from numpy import dot, array_equiv, inf, ndarray
import copy
import logging

class ExprEval(ABC):
    def __init__(self,expr):
        self.expr = expr

    def myExpr(self):
        return self.expr

    @abstractmethod
    def buildAForOrder(self,d):
        pass

    @abstractmethod
    def buildAllAUpToOrder(self,d):
        pass

    def buildF(self):
        pass 

    