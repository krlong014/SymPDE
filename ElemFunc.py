from Expr import UnaryExpr, BinaryExpr
import numpy as np


## Generic one-argument elementary function

class ElemFuncExpr(UnaryExpr):
    def __init__(self, name, arg):
        if ExprHelpers._shape(arg) != ScalarStructure():
            raise ValueError('ElemFuncExpr ctor: non-scalar arg [{}]'.format(arg))
        self.name = name
        self.arg = arg
        self.shape = arg.shape()

    def __str__(self):
        return '{}[{}]'.format(self.name, self.arg)

    def __repr__(self):
        return 'ElemFuncExpr[name={}, arg={}]'.format(self.name, self.arg)


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
