from abc import ABC, abstractmethod
from ExprShape import (ScalarShape, VectorShape)

class BasisBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


    @abstractmethod
    def shape(self):
        pass


class ScalarBasisBase:
    _shape = ScalarShape()

    def __init__(self):
        pass

    def shape(self):
        return ScalarBasisBase._shape

class VectorBasisBase:
    def __init__(self, shape):
        assert(isinstance(shape, VectorShape))
        self._shape = shape

    def shape(self):
        return self._shape
