from abc import ABC, abstractmethod
from ExprShape import (ScalarShape, VectorShape)

class BasisBase(ABC):
    def __init__(self, order, shape):
        self._order = order
        self._shape = shape

    def __str__(self):
        return '{}(order={})'.format(self._name(), self.order())

    def shape(self):
        return self._shape

    def order(self):
        return self._order

    def _name(self):
        return type(self).__name__

    #@abstractmethod
    def eval(self, pts, diffOp):
        pass


class ScalarBasisBase(BasisBase):

    def __init__(self, order):
        super().__init__(order, ScalarShape())

class VectorBasisBase(BasisBase):
    def __init__(self, order, shape):
        assert(isinstance(shape, VectorShape))
        super().__init__(order, shape)
