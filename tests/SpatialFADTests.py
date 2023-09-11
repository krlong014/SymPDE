import pytest
import numpy as np

from SymPDE.Expr import Expr
from SymPDE.ExprShape import VectorShape, ScalarShape
from SymPDE.DiffOp import (DiffOp, _Partial, Partial, _Div, Div, _Curl, Curl,
    _Rot, Rot, _Gradient, Gradient)
from SymPDE.ConstantExpr import ConstantScalarExpr, ConstantVectorExpr
from SymPDE.Coordinate import Coordinate
from SymPDE.VectorExpr import Vector
from SymPDE.AggExpr import AggExpr
from SymPDE.FunctionWithBasis import TestFunction
from SymPDE.BasisBase import BasisBase, VectorBasisBase, ScalarBasisBase
from SymPDE.SimpleEvaluator import compareEval, evalExpr, compareExprs
from SymPDE.UnivariateFunc import (UnivariateFuncExpr,
    Exp, Log, Sqrt, Cos, Sin, Tan, Cosh, Sinh, Tanh,
    ArcCos, ArcSin, ArcTan, ArcCosh, ArcSinh, ArcTanh, ArcTan2)
from SymPDE.SpatialFAD import spatialFAD


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
