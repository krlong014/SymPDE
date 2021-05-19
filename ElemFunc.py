from Expr import Expr, UnaryExpr, BinaryExpr, Coordinate
from ExprShape import ScalarShape


## Generic one-argument elementary function

class ElemFuncExpr(UnaryExpr):
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

    


##############################################################################
# The classic functions
##############################################################################

# Exp and log

def Exp(x):
    return ElemFuncExpr('Exp', x)

def Log(x):
    return ElemFuncExpr('Log', x)

# Square root

def Sqrt(x):
    return ElemFuncExpr('Sqrt', x)

# Trig functions

def Cos(x):
    return ElemFuncExpr('Cos', x)

def Sin(x):
    return ElemFuncExpr('Sin', x)

def Tan(x):
    return ElemFuncExpr('Tan', x)

# Hyperbolic functions

def Cosh(x):
    return ElemFuncExpr('Cosh', x)

def Sinh(x):
    return ElemFuncExpr('Sinh', x)

def Tanh(x):
    return ElemFuncExpr('Tanh', x)

# Inverse trig functions

def ArcCos(x):
    return ElemFuncExpr('ArcCos', x)

def ArcSin(x):
    return ElemFuncExpr('ArcSin', x)

def ArcTan(x):
    return ElemFuncExpr('ArcTan', x)

# Inverse hyperbolic functions

def ArcCosh(x):
    return ElemFuncExpr('ArcCosh', x)

def ArcSinh(x):
    return ElemFuncExpr('ArcSinh', x)

def ArcTanh(x):
    return ElemFuncExpr('ArcTanh', x)

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
