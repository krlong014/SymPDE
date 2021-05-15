from abc import ABC, abstractmethod
import numpy as np
from Expr import (Expr, BinaryExpr, SumExpr, ProductExpr, UnaryMinus,
                  QuotientExpr, PowerExpr, UnaryExpr, ConstantExprBase,
                  Coordinate)




class SimpleEvaluator(ABC):
    @abstractmethod
    def eval(self, varMap):
        pass

class UnaryEvaluator(SimpleEvaluator):
    def __init__(self, expr):
        assert(isinstance(expr, UnaryExpr))
        self.arg = makeEval(expr.arg)


class BinaryEvaluator(SimpleEvaluator):
    def __init__(self, expr):
        assert(isinstance(expr, BinaryExpr))
        self.L = makeEval(expr.L)
        self.R = makeEval(expr.R)

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


def makeEval(expr):
    evalMap = {
        ConstantExprBase : ConstantEvaluator,
        Coordinate : CoordinateEvaluator,
        SumExpr : SumEvaluator,
        ProductExpr : ProductEvaluator,
        QuotientExpr : QuotientEvaluator,
        PowerExpr : PowerEvaluator,
        UnaryMinus : UnaryMinusEvaluator
    }

    for k in evalMap.keys():
        if isinstance(expr, k):
            return evalMap[k](expr)

    raise ValueError('unknown expr type [{}]'.format(expr))

def evalRaw(varNames, varVals, exprString, constNames=[], constVals=[]):
    varMap = {}

    for name,val in zip(varNames, varVals):
        varMap[name]=val

    for name,val in zip(constNames, constVals):
        varMap[name]=val

    return eval(exprString, varMap)


def evalExpr(varNames, varVals, exprString, constNames=[], constVals=[]):

    varToExprMap = {}
    varToValMap = {}

    for i,(name,val) in enumerate(zip(varNames, varVals)):
        e = Coordinate(i, name)
        varToExprMap[name]=e
        varToValMap[e]=val

    for name,val in zip(constNames, constVals):
        varToExprMap[name]=val

    expr = eval(exprString, varToExprMap)
    evaluator = makeEval(expr)

    return evaluator.eval(varToValMap)

def compareEval(varNames, varVals, exprString, constNames={}, constVals={}):

    ex = evalExpr(varNames, varVals, exprString, constNames, constVals)
    raw = evalRaw(varNames, varVals, exprString, constNames, constVals)

    return (ex, raw)


if __name__=='__main__':

    print('raw=', evalRaw(('x', 'y'), (1.5, 0.25), '2*x*y+y/2'))
    print('ex=', evalExpr(('x', 'y'), (1.5, 0.25), '2*x*y+y/2'))

    print('comparing ', compareEval(('x', 'y'), (1.5, 0.25), '2*x*y+y/2'))
