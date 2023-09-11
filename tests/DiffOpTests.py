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
from SymPDE.SimpleEvaluator import compareEval, evalExpr


class TestDiffOpSanity:

    def test_Partial1(self):

        d_dx = _Partial(0)
        x = Coordinate(0)
        y = Coordinate(1)

        f = x*y
        df_dx = Partial(f, x)

        print('df_dx=', df_dx)
        assert(df_dx.sameas(DiffOp(d_dx, f)) and df_dx.shape()==f.shape())


    def test_Partial2(self):

        d_dx = _Partial(0)
        x = Coordinate(0)
        y = Coordinate(1)

        f = Vector(x*y, x+y)
        df_dx = Partial(f, x)

        print('df_dx=', df_dx)
        assert(df_dx.sameas(DiffOp(d_dx, f)) and df_dx.shape()==f.shape())


    def test_Div(self):

        x = Coordinate(0)
        y = Coordinate(1)

        div = _Div()
        F = Vector(x,y)
        divF = Div(F)

        print('Div(F)={}, shape={}'.format(divF, divF.shape()))
        assert(divF.sameas(DiffOp(div, F)) and divF.shape().sameas(ScalarShape()))


    def test_Curl(self):

        x = Coordinate(0)
        y = Coordinate(1)
        z = Coordinate(2)

        curl = _Curl()
        F = Vector(x,y,z)
        curlF = Curl(F)

        print('Curl(F)=', curlF)
        assert(curlF.sameas(DiffOp(curl, F)) and curlF.shape().dim()==3)


    def test_Rot(self):

        x = Coordinate(0)
        y = Coordinate(1)

        rot = _Rot()
        F = Vector(x,y)
        rotF = rot(F)

        print('Rot(F)=', rotF)
        assert(rotF.sameas(DiffOp(rot, F)) and rotF.shape().sameas(ScalarShape()))


class TestDiffOpOnFunction:

    def test_DiffOpOnTest1(self):

        basis = ScalarBasisBase(1)
        v = TestFunction(basis, 'v')
        dv_dx = Partial(v, 0)

        assert(dv_dx.isTest())


    def test_DivOnTest(self):

        basis = VectorBasisBase(1, VectorShape(2))
        V = TestFunction(basis, 'v')
        divV = Div(V)

        assert(divV.isTest())


    def test_CurlOnTest(self):

        basis = VectorBasisBase(1, VectorShape(3))
        V = TestFunction(basis, 'v')
        curlV = Curl(V)

        assert(curlV.isTest())


    def test_RotOnTest(self):

        basis = VectorBasisBase(1, VectorShape(2))
        V = TestFunction(basis, 'v')
        rotV = Rot(V)

        assert(rotV.isTest())


class TestDiffOpExpectedErrors:

    def test_DivOfScalar(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            bad = Div(x)


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))


    def test_CurlOfScalar(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            bad = Curl(x)


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))

    def test_RotOfScalar(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            bad = Rot(x)


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))


    def test_CurlOf2DVector(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            v = Vector(x,y)
            bad = Curl(v)


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))


    def test_RotOf3DVector(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            z = Coordinate(2)
            v = Vector(x,y,z)
            bad = Rot(v)


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))



    def test_DiffOpOfNonsense(self):
        with pytest.raises(TypeError) as err_info:
            bad = Rot('not an expr')


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))



    def test_DiffOpOfAgg(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            z = Coordinate(2)
            L = AggExpr(x,y,z)

            bad = Rot(L)


        print('detected expected exception: {}'.format(err_info))
        assert('cannot accept' in str(err_info.value))


    def test_NonsensePartial1(self):
        with pytest.raises(ValueError) as err_info:
            d_dx = _Partial(0)
            x = Coordinate(0)
            y = Coordinate(1)

            f = x*y
            df_dx = Partial(x, f)

        print('detected expected exception: {}'.format(err_info))
        assert('unable to interpret' in str(err_info.value))
