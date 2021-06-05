from ExprShape import ScalarShape, VectorShape, TensorShape
from Expr import (Expr, UnaryMinus, SumExpr, ProductExpr, CrossProductExpr,
                    QuotientExpr, PowerExpr, Coordinate, AggExpr,
                    VectorExprInterface)
from DiffOp import (DiffOp, HungryDiffOp, Partial, _Partial, _IdentityOp,
    _Div, Div, _Gradient, Gradient, _Curl, Curl, _Rot, Rot)
from UnivariateFunc import (UnivariateFuncExpr,
    Exp, Log, Sqrt, Cos, Sin, Tan, Cosh, Sinh, Tanh,
    ArcCos, ArcSin, ArcTan, ArcCosh, ArcSinh, ArcTanh, ArcTan2)
from VectorExprs import Vector
from SimpleEvaluator import compareExprs

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




def spatialFAD(expr, op=_IdentityOp()):

    # Make sure arguments are valid
    assert(isinstance(expr, Expr) and isinstance(op, HungryDiffOp))
    assert(not isinstance(expr, AggExpr))



    # If identity op, simply evaluate the expr
    if isinstance(expr, DiffOp):
        return spatialFAD(expr.arg(), expr.op())

    if isinstance(op, _IdentityOp):
        return expr

    if expr.isSpatialConstant():
        if expr.isScalar():
            if isinstance(op, _Gradient):
                return Vector([0,]*op.dim())
            if isinstance(op, (_Curl, _Div, _Rot)):
                raise SpatialFADException('Vector operator applied to scalar',
                    op, expr)
            return 0
        if expr.isVector():
            if isinstance(op, _Div, _Rot):
                return 0
            return Vector([0,]*expr.shape().dim())
        raise SpatialFADException('tensor ops not yet implemented', op)

    if isinstance(expr, SumExpr):
        return spatialFAD(expr.left(), op) + expr.sign*spatialFAD(expr.right(), op)

    if isinstance(expr, UnaryMinus):
        return -spatialFAD(expr.arg(), op)

    if isinstance(expr, ProductExpr):
        return differentiateProduct(expr, op)

    if isinstance(expr, CrossProductExpr):
        return differentiateCrossProduct(expr, op)

    if isinstance(expr, PowerExpr):
        return differentiatePower(expr, op)

    if isinstance(expr, QuotientExpr):
        return differentiateQuotient(expr, op)

    if isinstance(expr, Coordinate):
        return differentiateCoordinate(expr, op)

    # if isinstance(expr, VectorExprInterface):
    #     return differentiateVector(expr, op)

    if isinstance(expr, UnivariateFuncExpr):
        return differentiateUnivariateFunc(expr, op)

    if isinstance(expr, DiffOp):
        return spatialFAD(expr.arg(), expr.op())

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

    return e*b**(e-1)*spatialFAD(b,op) + Log(b)*expr*spatialFAD(e, op)


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


if __name__=='__main__':

    x = Coordinate(0)
    y = Coordinate(1)
    z = Coordinate(2)

    f = x*x*x

    df = spatialFAD(Partial(f,x))
    print('df=', df)


class TestDeriv1D:

    def test_DiffConstant(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = 4.0
        df0 = 0.0
        df = spatialFAD(Partial(f, x))

        assert(compareExprs(df, df0, varMap))


    def test_DiffCoord(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = x
        df0 = 1.0
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))


    def test_DiffPower1(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = x**3
        df0 = 3*x**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))


    def test_DiffPower2(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = x*(x**2)
        df0 = 3*x**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))


    def test_DiffPower3(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = x*x*x
        df0 = 3*x**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))


    def test_DiffPower4(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = 2**x
        df0 = f*Log(2)
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))


    def test_DiffPower5(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = x**x
        df0 = f*(1+Log(x))
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))


    def test_DiffSum1(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = 1+5*x
        df0 = 5
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffSum2(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = 1-5*x
        df0 = -5
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffPoly1(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        f = 4.0 - 2.0*x + 3.0*x**2
        df0 = -2.0 + 6.0*x
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffPoly2(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        q = 4.0 - 2.0*x + 3.0*x**2
        f = q*q
        df0 = 2*(-2.0 + 6.0*x)*q
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffPoly3(self):
        x = Coordinate(0)
        varMap = { x:1.5 }
        q = 4.0 - 2.0*x + 3.0*x**2
        f = q**3
        df0 = 3*(-2.0 + 6.0*x)*q**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffQuotient1(self):
        x = Coordinate(0)
        varMap = { x:1.5 }

        f = 1/(1+x)
        df0 = -1/(1+x)**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffQuotient2(self):
        x = Coordinate(0)
        varMap = { x:1.5 }

        f = x/(1+x)
        df0 = 1/(1+x)**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffQuotient3(self):
        x = Coordinate(0)
        varMap = { x:1.5 }

        f = 1/(1+x**2)
        df0 = -2*x/(1+x**2)**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))

    def test_DiffQuotient4(self):
        x = Coordinate(0)
        varMap = { x:1.5 }

        f = x/(1+x**2)
        df0 = (1-x**2)/(1+x**2)**2
        df = spatialFAD(Partial(f, x))
        print('df={}'.format(df))

        assert(compareExprs(df, df0, varMap))
