from abc import ABC, abstractmethod
from functools import total_ordering
from . ExprShape import (ScalarShape, TensorShape, VectorShape)

@total_ordering
class HungryDiffOp(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def acceptShape(self, input):
        pass

    @abstractmethod
    def outputShape(self, input):
        pass

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __call__(self, arg):
        
        # We'll need some expression classes here. Import within this function
        # definition instead of at the global scope
        from . Expr import Expr
        from . DiffOp import DiffOp, DiffOpOnFunction
        from . FunctionWithBasis import FunctionWithBasis

        # Make sure the type makes sense
        if not Expr._convertibleToExpr(arg):
            raise TypeError(
                'diff op [{}] cannot accept type [{}]'.format(self, arg)
                )

        f = Expr._convertToExpr(arg)

        # Make sure the operator and input are consistent
        if not self.acceptShape(f.shape()):
            raise TypeError(
                'diff op [{}] cannot accept argument [{}]'.format(self,f)
                )


        # Form the diff op expression
        if isinstance(f, FunctionWithBasis):
            return DiffOpOnFunction(self, f)
        return DiffOp(self, f)


class _IdentityOp(HungryDiffOp):
    def __init__(self):
        super().__init__()

    def __call__(self, arg):
        return arg

    def acceptShape(self, input):
        return True

    def outputShape(self, input):
        return input

    def __str__(self):
        return 'IdentityOp'

class _Partial(HungryDiffOp):
    def __init__(self, dir, name=None):
        super().__init__()
        self._dir = dir
        if name==None:
            from . Expr import Expr
            self._name = Expr._dirName(dir)
        else:
            self._name = name

    def acceptShape(self, input):
        return True

    def outputShape(self, input):
        return input

    def direction(self):
        return self._dir

    def __str__(self):
        return 'd_d{}'.format(self._name)

def Partial(f, coord):
    from . Coordinate import Coordinate

    if isinstance(coord, int):
        op = _Partial(coord)
    elif isinstance(coord, Coordinate):
        op = _Partial(coord.direction())
    else:
        raise ValueError('unable to interpret direction {} in Partial()'.format(coord))

    return op(f)


class _Div(HungryDiffOp):

    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return not isinstance(input, ScalarShape)

    def outputShape(self, input):
        if isinstance(input, (VectorShape, VectorShape)):
            return ScalarShape()
        if isinstance(input, TensorShape):
            return VectorShape(input.dim())
        raise ValueError(
            'Div.outputShape expected vector or tensor, got [{}]'.format(input)
            )
    def __str__(self):
        return 'Div'


def Div(f):
    div = _Div()
    return div(f)


class _Gradient(HungryDiffOp):
    def __init__(self, dim):
        super().__init__()
        self._dim=dim

    def dim(self):
        return self._dim

    def acceptShape(self, input):
        return not isinstance(input, TensorShape)

    def outputShape(self, input):
        if isinstance(input, ScalarShape):
            return VectorShape(self.dim())
        if isinstance(input, VectorShape):
            return TensorShape(self.dim())
        raise ValueError('grad.outputShape expected scalar or vector, got [{}]'.format(input))

    def __str__(self):
        return 'Grad'


def Gradient(f, dim=3):
    grad = _Gradient(dim)
    return grad(f)


class _Curl(HungryDiffOp):
    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return isinstance(input, VectorShape) and input.dim()==3

    def outputShape(self, input):
        assert(self.acceptShape(input))
        return input

    def __str__(self):
        return 'Curl'


def Curl(f):
    curl = _Curl()
    return curl(f)


class _Rot(HungryDiffOp):
    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return isinstance(input, VectorShape) and input.dim()==2

    def outputShape(self, input):
        assert(self.acceptShape(input))
        return ScalarShape()

    def __str__(self):
        return 'Rot'



def Rot(f):
    rot = _Rot()
    return rot(f)
