from Expr import (Expr, BinaryExpr, SumExpr, ProductExpr, UnaryMinus,
                  DotProductExpr, CrossProductExpr,
                  QuotientExpr, PowerExpr, UnaryExpr, ConstantScalarExpr,
                  ConstantVectorExpr, Coordinate)
from UnivariateFunc import (UnivariateFuncExpr, ArcTan2Func)


class EvaluatorNode:
    def __init__(self, expr):
        self._expr = expr

    def expr(self):
        return self._expr

    @singledispatchmethod
    def makeNode(self, arg):
        raise NotImplementedError('unknown type in makeNode')

    @makeNode.register
    def _(self, arg:SumExpr):
        return SumEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:ProductExpr):
        return ProductEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:UnaryMinus):
        return UnaryMinusEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:DotProductExpr):
        return DotProductEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:CrossProductExpr):
        return CrossProductEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:QuotientExpr):
        return QuotientEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:PowerExpr):
        return PowerEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:Coordinate):
        return CoordinateEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:ArcTan2Func):
        return ArcTan2EvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:UnivariateFuncExpr):
        return UnivariateFuncEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:ConstantVectorExpr):
        return ConstantVectorEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:ConstantScalarExpr):
        return ConstantScalarEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:FunctionWithScalarBasis):
        return ScalarFunctionEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:FunctionWithVectorBasis):
        return VectorFunctionEvaluatorNode(arg)

    @makeNode.register
    def _(self, arg:DiffOpOnFunction):
        return DiffOpOnFunctionEvaluatorNode(arg)

class UnaryEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, UnaryExpr))
        self._arg = self.makeNode(expr.arg())

    def arg(self):
        return self._arg

class BinaryEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, BinaryExpr))
        self._left = self.makeNode(expr.left())
        self._right = self.makeNode(expr.right())

    def left(self):
        return self._left

    def right(self):
        return self._right

class SumEvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, SumExpr))

class ProductEvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, ProductExpr))

class DotProductEvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, DotProductExpr))

class CrossProductEvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, CrossProductExpr))

class QuotientEvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, QuotientExpr))

class PowerEvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, PowerExpr))

class UnaryMinusEvaluatorNode(UnaryEvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, UnaryMinus))

class ConstantVectorEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, ConstantVectorExpr))

class ConstantScalarEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, ConstantExprBase))

class CoordinateEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        super().__init__(expr)
        assert(isinstance(expr, Coordinate))

class UnivariateFuncEvaluatorNode(UnaryEvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, UnivariateFuncExpr))
        super().__init__(expr)

class ArcTan2EvaluatorNode(BinaryEvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, ArcTan2Func))
        super().__init__(expr)

class SymbFunctionEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, FunctionWithBasis))
        super().__init__(expr))

class ScalarSymbFunctionEvaluatorNode(SymbFunctionEvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, FunctionWithScalarBasis))
        super().__init__(expr)

class VectorSymbFunctionEvaluatorNode(SymbFunctionEvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, FunctionWithVectorBasis))
        super().__init__(expr))

class DiffOpOnFunctionEvaluatorNode(EvaluatorNode):
    def __init__(self, expr):
        assert(isinstance(expr, DiffOpOnFunction))
        super().__init__(expr)
