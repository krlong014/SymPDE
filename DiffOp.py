from abc import ABC, abstractmethod
from Expr import Expr, UnaryExpr
from ExprShape import (ExprShape, ScalarShape, TensorShape, RowVectorShape,
                        ColumnVectorShape, VectorShape)

class DiffOp(UnaryExpr):

    def __init__(self, op, func):
        super().__init__(self, arg)
        self.op = op
        self.myShape = op.outputShape(op.shape())

    def __str__(self):
        return '{}({})'.format(self.op.__str__(), self.op.__str__())

    def shape(self):
        return self.myShape


class HungryDiffOp(Expr, ABC):

    def __init__(self):
        pass

    @abstractmethod
    def acceptShape(self, input):
        pass

    @abstractmethod
    def outputShape(self, input):
        pass


    def __call__(self, arg):

        # Make sure the type makes sense
        if not instanceof(arg, (Expr, int, float)):
            raise TypeError(
                'diff op [{}] cannot accept type [{}]'.format(self, arg)
                )

        # Make sure the operator and input are consistent
        if not self.acceptShape(arg.shape()):
            raise TypeError(
                'diff op [{}] cannot accept argument [{}]'.format(self,arg)
                )

        # Derivatives of constants are zero
        if Expr._isConstant(arg):
            return 0

        # Form the diff op expression
        return DiffOp(self, op, arg)


class Partial(HungryDiffOp):
    def __init__(self, dir, name=None):
        self.dir = dir
        if name==None:
            self.myName = _dirName(dir)
        else:
            self.myName = name


    def acceptShape(self, input):
        return True

    def outputShape(self, input):
        return input.shape()

    def __str__(self):
        return 'd{}'.format(self.myName)



class Div(HungryDiffOp):

    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return not isinstance(input, ScalarShape)

    def outputShape(self, input):
        if isinstance(input, (RowVectorShape, ColumnVectorShape)):
            return ScalarShape()
        if isinstance(input, TensorShape):
            return ColumnVectorShape(input.dim)
        raise ValueError(
            'Div.outputShape expected vector or tensor, got [{}]'.format(input)
            )
    def __str__(self):
        return 'Div'

class Gradient(HungryDiffOp):
    def __init__(self, dim):
        self.dim=dim

    def acceptShape(self, input):
        return not isinstance(input, TensorShape)

    def outputShape(self, input):
        if isinstance(input, ScalarShape):
            return ColumnVectorShape(self.dim)
        if isinstance(input, (RowVectorShape, ColumnVectorShape)):
            return TensorShape(self.dim)
        raise ValueError('Div.outputShape expected vector or tensor, got [{}]'.format(input))


    def __str__(self):
        return 'Grad'

class Curl(HungryDiffOp):
    def __init__(self):

    def __str__(self):
        return 'Curl'

class Rot(HungryDiffOp):
    def __init__(self):

    def __str__(self):
        return 'Rot'
