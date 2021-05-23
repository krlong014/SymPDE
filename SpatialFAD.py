from ExprShape import ScalarShape, VectorShape, TensorShape
from Expr import (Expr, UnaryMinus, SumExpr, ProductExpr, CrossProductExpr,
                    QuotientExpr, PowerExpr, Coordinate, AggExpr)
from DiffOp import (DiffOp, HungryDiffOp, Partial, _Partial,
    _Div, Div, _Gradient, Gradient, _Curl, Curl, _Rot, Rot)

class SpatialFADException(Exception):
    def __init__(self, descr, op, *args):
        super().__init__()
        self._op = op
        self._args = args
        self._descr

    def __str__(self):
        str = 'Spatial FAD exception [{}]:\n op=[{}]\n'.format(self._descr,self._op)

        if len(self._args)>1:
            str = str + 'args='

        if len(self._args)==1:
            str = str + 'arg='

        for i,a in enumerate(self._args):
            if i>0:
                str = str + ', '
            str = str + '[{}]'.format(a)

        return str




def spatialFAD(expr, op):

    # Make sure arguments are valid
    assert(isinstance(expr, Expr) and isinstance(op, HungryDiffOp))
    assert(not isinstance(expr, AggExpr))

    if expr.isSpatialConstant():
        if expr.isScalar():
            return 0
        if expr.isVector():
            return Vector([0,]*expr.shape().dim())
        raise SpatialFADException('tensor ops not yet implemented', op)

    if isinstance(expr, SumExpr):
        return spatialDiff(expr.L, op) + spatialDiff(expr.R, op)

    if isinstance(expr, UnaryMinus):
        return -spatialDiff(expr, op)

    if isinstance(expr, ProductExpr):
        return differentiateProduct(expr, op)

    if isinstance(expr, CrossProductExpr):
        return differentiateCrossProduct(expr, op)

    if isinstance(expr, PowerExpr):
        return differentiatePower(expr, op)

    if isinstance(expr, QuotientExpr):
        return differentiateQuotient(expr, op)

    if isinstance(expr, CoordinateExpr):
        return differentiateCoordinate(expr, op)

    if isinstance(expr, VectorExprInterface):
        return differentiateVector(expr, op)

    if isinstance(expr, UnivariateFuncExpr):
        return differentiateUnivariateFunc(expr, op)

    raise SpatialFADException('no rule to differentiate', op, expr)


def differentiateProduct(expr, op):
    assert(isinstance(expr, ProductExpr))

    left = expr.left()
    right = expr.right()

    # Partial derivative:
    # partial(a*b) = partial(a)*b + a*partial(b)
    if isinstance(op, _Partial):
        return left*spatialFAD(right, op) + spatialFAD(left, op)*right

    # Gradient:
    # grad(a*b) = grad(a)*b + a*grad(b)
    if isinstance(op, _Gradient):
        if left.isScalar() and right.isScalar():
            return left*spatialFAD(right, op) + spatialFAD(left, op)*right
        raise SpatialFADException('gradient product rule',
            op, (left,right))

    # Divergence:
    # Div(s*v) = s * div(v) + grad(s)*v
    if isinstance(op, _Div):
        if left.isVector() and right.isScalar():
            v = left
            s = right
        elif left.isScalar() and right.isVector():
            v = right
            s = left
        else:
            raise SpatialFADException('divergence product rule',
                op, (left,right))

        vecDim = v.shape().dim()
        grad = _Gradient(vecDim)
        return s*spatialFAD(v, op) + spatialFAD(s, grad)*v


    # Curl and Rot:
    # Curl(s*v) = s * curl(v) + cross(grad(s), v)
    if isinstance(op, _Curl) or isinstance(op, _Rot):
        if left.isVector() and right.isScalar():
            v = left
            s = right
        elif left.isScalar() and right.isVector():
            v = right
            s = left
        else:
            raise SpatialFADException('curl product rule',
                op, (left,right))

        vecDim = v.shape().dim()
        grad = _Gradient(vecDim)
        return s*spatialFAD(v, op) + Cross(spatialFAD(s, grad), v)

    raise SpatialFADException('product rule case not handled', op, (left,right))



# To-do
def differentiateCrossProduct(expr, op):
    assert(isinstance(expr, CrossProductExpr))
    raise SpatialFADException('Product rule for cross product not implemented')

def differentiateQuotient(expr, op):
    assert(isinstance(expr, QuotientExpr))

    num = expr.left()
    den = expr.right()

    assert(den.isScalar())

    if num.isScalar() and den.isScalar() and isinstance(op, (_Partial, _Gradient)):
        return (den*spatialFAD(num, op) - num*spatialFAD(den, op))/den**2

    # To do:
    # - div(v/s)
    # - curl(v/s) and rot(v/s)

    raise SpatialFADException('product rule case not handled', op, (left,right))



def differentiateCoordinate(expr, op):
    assert(isinstance(expr, Coordinate))
    assert(isinstance(op, (_Partial, _Gradient)))

    if isinstance(op, _Partial):
        if op.direction() == expr.direction():
            return 1
        else:
            return 0

    if isinstance(op, _Gradient):
        e_i = [0,]*op.dim()
        e_i[expr.direction()]=1
        return Vector(e_i)


def differentiatePower(expr, op):
    assert(isinstance(expr, PowerExpr))
    assert(isinstance(op, (_Partial, _Gradient)))

    b = expr.left()
    e = expr.right()

    assert(b.isScalar())
    assert(e.isScalar())

    return b**(e-1)*spatialFAD(b,op) + Log(b)*expr*spatialFAD(e, op)


def differentiateVector(expr, op):
    assert(isinstance(expr, VectorExprBase))
    assert(isinstance(op, (_Partial, _Div, _Curl, _Rot)))

    raise NotImplementedError('differentiateVector')


def differentiateUnivariateFunc(expr, op):
    assert(isinstance(expr, UnivariateFuncExpr))
    assert(isinstance(op, (_Partial, _Gradient)))

    # Differentiate f(u):
    # Partial(f(u)) = f'(u) * Partial(u)
    # grad(f(u)) = f'(u) * grad(u)
    f = expr
    u = expr.arg()
    return f.deriv(u) * spatialFAD(u, op)
