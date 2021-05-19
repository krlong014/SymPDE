from abc import ABC, abstractmethod
from Expr import Expr, UnaryExpr, BinaryExpr, Coordinate
from ExprShape import ScalarShape


## Generic one-argument elementary function

class ElemFuncExpr(UnaryExpr, ABC):
    def __init__(self, name, arg):
        assert(Expr._convertibleToExpr(arg))
        expr = Expr._convertToExpr(arg)
        if not isinstance(expr.shape(), ScalarShape):
            raise ValueError('ElemFuncExpr ctor: non-scalar arg [{}]'.format(expr))
        super().__init__(expr, expr.shape())
        self._name = name

    def name(self):
        return self._name


    def __str__(self):
        return '{}[{}]'.format(self._name, self.arg())

    def __repr__(self):
        return 'ElemFuncExpr[name={}, arg={}]'.format(self._name, self.arg())

    @abstractmethod
    def deriv(self, x):
        pass





# The two-argument arctangent arctan2(y,x)
class ArcTan2Func(BinaryExpr):
    def __init__(self, y, x):
        super().__init__(y, x, ScalarShape())

    def __str__(self):
        return 'ArcTan2({},{})'.format(self.y(), self.x())

    def x(self):
        return self.right()

    def y(self):
        return self.left()

def ArcTan2(y, x):
    return ArcTan2Func(y, x)


##############################################################################
# The classic functions
##############################################################################


class ExpFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Exp', x)

    def deriv(self, x):
        return Exp(x)


def Exp(x):
    return ExpFunc(x)


class LogFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Log', x)

    def deriv(self, x):
        return 1/x


def Log(x):
    return LogFunc(x)


class SqrtFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Sqrt', x)

    def deriv(self, x):
        return 1/2/Sqrt(x)


def Sqrt(x):
    return SqrtFunc(x)


class CosFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Cos', x)

    def deriv(self, x):
        return -Sin(x)


def Cos(x):
    return CosFunc(x)


class SinFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Sin', x)

    def deriv(self, x):
        return Cos(x)


def Sin(x):
    return SinFunc(x)


class TanFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Tan', x)

    def deriv(self, x):
        return 1/Cos(x)**2


def Tan(x):
    return TanFunc(x)


class CoshFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Cosh', x)

    def deriv(self, x):
        return Sinh(x)


def Cosh(x):
    return CoshFunc(x)


class SinhFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Sinh', x)

    def deriv(self, x):
        return Cosh(x)


def Sinh(x):
    return SinhFunc(x)


class TanhFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('Tanh', x)

    def deriv(self, x):
        return 1/Cosh(x)**2


def Tanh(x):
    return TanhFunc(x)


class ArcCosFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('ArcCos', x)

    def deriv(self, x):
        return -1/Sqrt(1-x**2)


def ArcCos(x):
    return ArcCosFunc(x)


class ArcSinFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('ArcSin', x)

    def deriv(self, x):
        return 1/Sqrt(1-x**2)


def ArcSin(x):
    return ArcSinFunc(x)


class ArcTanFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('ArcTan', x)

    def deriv(self, x):
        return 1/(1+x**2)


def ArcTan(x):
    return ArcTanFunc(x)


class ArcCoshFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('ArcCosh', x)

    def deriv(self, x):
        return 1/Sqrt(x**2-1)


def ArcCosh(x):
    return ArcCoshFunc(x)


class ArcSinhFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('ArcSinh', x)

    def deriv(self, x):
        return 1/Sqrt(1+x**2)


def ArcSinh(x):
    return ArcSinhFunc(x)


class ArcTanhFunc(ElemFuncExpr):
    def __init__(self, x):
        super().__init__('ArcTanh', x)

    def deriv(self, x):
        return 1/(1-x**2)


def ArcTanh(x):
    return ArcTanhFunc(x)
