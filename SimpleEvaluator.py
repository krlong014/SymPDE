from abc import ABC, abstractmethod
from numpy import exp, sqrt
import numpy as np
from Expr import (Expr, BinaryExpr, SumExpr, ProductExpr, UnaryMinus,
                  DotProductExpr, CrossProductExpr,
                  QuotientExpr, PowerExpr, UnaryExpr, ConstantExprBase,
                  Coordinate)
from UnivariateFunc import (UnivariateFuncExpr, Exp, Log, Sqrt, ArcTan2Func, ArcTan2,
                      Cos, Sin, Tan, Cosh, Sinh, Tanh,
                      ArcCos, ArcSin, ArcTan, ArcCosh, ArcSinh, ArcTanh)





class SimpleEvaluator(ABC):
    @abstractmethod
    def eval(self, varMap):
        pass

class UnaryEvaluator(SimpleEvaluator):
    def __init__(self, expr):
        assert(isinstance(expr, UnaryExpr))
        self.arg = makeEval(expr.arg())


class BinaryEvaluator(SimpleEvaluator):
    def __init__(self, expr):
        assert(isinstance(expr, BinaryExpr))
        self.L = makeEval(expr.left())
        self.R = makeEval(expr.right())

class SumEvaluator(BinaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, SumExpr))
        self.sign = expr.sign

    def eval(self, varMap):
        if self.sign==1:
            return self.L.eval(varMap) + self.R.eval(varMap)
        else:
            return self.L.eval(varMap) - self.R.eval(varMap)

class ProductEvaluator(BinaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, ProductExpr))

    def eval(self, varMap):
        return self.L.eval(varMap) * self.R.eval(varMap)

class DotProductEvaluator(BinaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, DotProductExpr))

    def eval(self, varMap):
        return np.dot(self.L.eval(varMap), self.R.eval(varMap))

class CrossProductEvaluator(BinaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, CrossProductExpr))

    def eval(self, varMap):
        return np.cross(self.L.eval(varMap), self.R.eval(varMap))

class QuotientEvaluator(BinaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, QuotientExpr))

    def eval(self, varMap):
        return self.L.eval(varMap) / self.R.eval(varMap)

class PowerEvaluator(BinaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, PowerExpr))

    def eval(self, varMap):
        return np.power(self.L.eval(varMap), self.R.eval(varMap))

class UnaryMinusEvaluator(UnaryEvaluator):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, UnaryMinus))

    def eval(self, varMap):
        return -self.arg.eval(varMap)


class ConstantEvaluator(SimpleEvaluator):
    def __init__(self, expr):
        self.data = expr.data()
        assert(isinstance(expr, ConstantExprBase))


    def eval(self, varMap):
        return self.data

class CoordinateEvaluator(SimpleEvaluator):
    def __init__(self, expr):
        assert(isinstance(expr, Coordinate))
        self.coord = expr

    def eval(self, varMap):
        return varMap[self.coord]

class UnivariateFuncEvaluator(UnaryEvaluator):
    funcMap = {
        'Exp': np.exp,
        'Log': np.log,
        'Sqrt': np.sqrt,
        'Cos': np.cos,
        'Sin': np.sin,
        'Tan': np.tan,
        'Cosh': np.cosh,
        'Sinh': np.sinh,
        'Tanh': np.tanh,
        'ArcCos': np.arccos,
        'ArcSin': np.arcsin,
        'ArcTan': np.arctan,
        'ArcCosh': np.arccosh,
        'ArcSinh': np.arcsinh,
        'ArcTanh': np.arctanh,
        'ArcTan2' : np.arctan2
    }

    def __init__(self, expr):
        assert(isinstance(expr, UnivariateFuncExpr))
        super().__init__(expr)
        self._func = UnivariateFuncEvaluator.funcMap[expr.name()]

    def eval(self, varMap):
        return self._func(self.arg.eval(varMap))

class ArcTan2Evaluator(BinaryEvaluator):
    def __init__(self, expr):
        assert(isinstance(expr, ArcTan2Func))
        super().__init__(expr)

    def eval(self, varMap):
        yVal = self.L.eval(varMap)
        xVal = self.R.eval(varMap)
        return np.arctan2(yVal, xVal)


def makeEval(expr):
    evalMap = {
        ConstantExprBase : ConstantEvaluator,
        Coordinate : CoordinateEvaluator,
        SumExpr : SumEvaluator,
        ProductExpr : ProductEvaluator,
        DotProductExpr : DotProductEvaluator,
        CrossProductExpr : CrossProductEvaluator,
        QuotientExpr : QuotientEvaluator,
        PowerExpr : PowerEvaluator,
        UnaryMinus : UnaryMinusEvaluator,
        UnivariateFuncExpr : UnivariateFuncEvaluator,
        ArcTan2Func : ArcTan2Evaluator
    }

    for k in evalMap.keys():
        if isinstance(expr, k):
            return evalMap[k](expr)

    raise ValueError('unknown expr type [{}]'.format(expr))

def evalRaw(varNames, varVals, exprString, constNames=[], constVals=[]):

    varMap = UnivariateFuncEvaluator.funcMap

    for name,val in zip(varNames, varVals):
        varMap[name]=val

    for name,val in zip(constNames, constVals):
        varMap[name]=val

    return eval(exprString, varMap)


def evalExpr(varNames, varVals, exprString, constNames=[], constVals=[]):

    varToExprMap = {'Exp' : Exp,
        'Sqrt' : Sqrt,
        'Log' : Log,
        'Cos' : Cos,
        'Sin' : Sin,
        'Tan' : Tan,
        'Cosh' : Cosh,
        'Sinh' : Sinh,
        'Tanh' : Tanh,
        'ArcCos' : ArcCos,
        'ArcSin' : ArcSin,
        'ArcTan' : ArcTan,
        'ArcCosh' : ArcCosh,
        'ArcSinh' : ArcSinh,
        'ArcTanh' : ArcTanh,
        'ArcTan2' : ArcTan2}
    varToValMap = {}

    for i,(name,val) in enumerate(zip(varNames, varVals)):
        e = Coordinate(i, name)
        varToExprMap[name]=e
        varToValMap[e]=val

    for name,val in zip(constNames, constVals):
        varToExprMap[name]=val

    #print('expr to eval: {}'.format(exprString), flush=True)
    #print('var to expr map: {}'.format(varToExprMap), flush=True)
    expr = eval(exprString, varToExprMap)
    evaluator = makeEval(expr)

    return evaluator.eval(varToValMap)

def compareEval(varNames, varVals, exprString, constNames={}, constVals={}):

    ex = evalExpr(varNames, varVals, exprString, constNames, constVals)
    raw = evalRaw(varNames, varVals, exprString, constNames, constVals)

    return (ex, raw)
