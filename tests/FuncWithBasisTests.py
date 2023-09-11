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
from SymPDE.FunctionWithBasis import (TestFunction, UnknownFunction,
    DiscreteFunction)
from SymPDE.BasisBase import BasisBase, VectorBasisBase, ScalarBasisBase
from SymPDE.SimpleEvaluator import compareEval, evalExpr, compareExprs
from SymPDE.UnivariateFunc import (UnivariateFuncExpr,
    Exp, Log, Sqrt, Cos, Sin, Tan, Cosh, Sinh, Tanh,
    ArcCos, ArcSin, ArcTan, ArcCosh, ArcSinh, ArcTanh, ArcTan2)
from SymPDE.DiscreteSpaceBase import DiscreteSpaceBase
from SymPDE.SpatialFAD import spatialFAD


class TestFunctionIdentification:

    def test_TestFunc(self):
        v = TestFunction(ScalarBasisBase(0), 'v')
        V = TestFunction(VectorBasisBase(0,VectorShape(2)), 'V')

        v0 = V[0]
        v1 = V[1]

        vIsTest = v.isTest()
        VIsTest = V.isTest()
        v0IsTest = v0.isTest()
        v1IsTest = v1.isTest()

        vIsNotUnknown = not v.isUnknown()
        VIsNotUnknown = not V.isUnknown()
        v0IsNotUnknown = not v0.isUnknown()
        v1IsNotUnknown = not v1.isUnknown()

        v0IndexIsZero = v0.index()==0
        v1IndexIsOne = v1.index()==1

        assert(vIsTest and VIsTest and v0IsTest and v1IsTest
            and vIsNotUnknown and VIsNotUnknown
            and v0IsNotUnknown and v1IsNotUnknown
            and v0IndexIsZero and v1IndexIsOne)

    def test_UnknownFunc(self):
        v = UnknownFunction(ScalarBasisBase(0), 'v')
        V = UnknownFunction(VectorBasisBase(0,VectorShape(2)), 'V')

        v0 = V[0]
        v1 = V[1]

        vIsUnknown = v.isUnknown()
        VIsUnknown = V.isUnknown()
        v0IsUnknown = v0.isUnknown()
        v1IsUnknown = v1.isUnknown()

        vIsNotTest = not v.isTest()
        VIsNotTest = not V.isTest()
        v0IsNotTest = not v0.isTest()
        v1IsNotTest = not v1.isTest()

        v0IndexIsZero = v0.index()==0
        v1IndexIsOne = v1.index()==1

        assert(vIsUnknown and VIsUnknown and v0IsUnknown and v1IsUnknown
            and vIsNotTest and VIsNotTest
            and v0IsNotTest and v1IsNotTest
            and v0IndexIsZero and v1IndexIsOne)

    def test_DiscreteFunc(self):
        v = DiscreteFunction(DiscreteSpaceBase(ScalarBasisBase(0)), 'v')
        V = DiscreteFunction(DiscreteSpaceBase(VectorBasisBase(0,VectorShape(2))), 'V')

        v0 = V[0]
        v1 = V[1]

        vIsDiscrete = v.isDiscrete()
        VIsDiscrete = V.isDiscrete()
        v0IsDiscrete = v0.isDiscrete()
        v1IsDiscrete = v1.isDiscrete()

        assert(vIsDiscrete and VIsDiscrete and v0IsDiscrete and v1IsDiscrete)
