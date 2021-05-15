import pytest
import numpy as np
from Expr import (Expr, ConstantScalarExpr, ConstantVectorExpr,
    Coordinate)
from SimpleEvaluator import compareEval, evalExpr


class TestOpsWithConstants:

    # constant plus constant
    def test_ConstantPlusConstant(self):
        ex = ConstantScalarExpr(1)+ConstantScalarExpr(2)
        assert(ex.data()==3)

    # 0+constant and constant+0
    def test_ConstantPlusZero1(self):
        ex = ConstantScalarExpr(1)+ConstantScalarExpr(0)
        assert(ex ==ConstantScalarExpr(1))

    def test_ConstantPlusZero2(self):
        ex = ConstantScalarExpr(1)+0
        assert(ex == ConstantScalarExpr(1))

    def test_ZeroPlusConstant1(self):
        ex = ConstantScalarExpr(0)+ConstantScalarExpr(1)
        assert(ex == ConstantScalarExpr(1))

    def test_ZeroPlusConstant2(self):
        ex = 0+ConstantScalarExpr(1)
        assert(ex == ConstantScalarExpr(1))

    # constant minus constant
    def test_ConstantMinusConstant(self):
        ex = ConstantScalarExpr(1)-ConstantScalarExpr(2)
        assert(ex.data()==-1)

    # 0-constant and constant-0
    def test_ConstantMinusZero1(self):
        ex = ConstantScalarExpr(1)-ConstantScalarExpr(0)
        assert(ex == ConstantScalarExpr(1))

    def test_ConstantMinusZero2(self):
        ex = ConstantScalarExpr(1)-0
        assert(ex == ConstantScalarExpr(1))

    def test_ZeroMinusConstant1(self):
        ex = ConstantScalarExpr(0)-ConstantScalarExpr(1)
        assert(ex == ConstantScalarExpr(-1))

    def test_ZeroMinusConstant2(self):
        ex = 0-ConstantScalarExpr(1)
        assert(ex == ConstantScalarExpr(-1))


    # constant times constant
    def test_ConstantTimesConstant(self):
        ex = ConstantScalarExpr(3)*ConstantScalarExpr(2)
        assert(ex.data()==6)


    # 1*constant and constant*1
    def test_ConstantTimesOne1(self):
        ex = ConstantScalarExpr(2)*ConstantScalarExpr(1)
        assert(ex == ConstantScalarExpr(2))

    def test_ConstantTimesOne2(self):
        ex = ConstantScalarExpr(2)*1
        assert(ex == ConstantScalarExpr(2))

    def test_OneTimesConstant1(self):
        ex = ConstantScalarExpr(1)*ConstantScalarExpr(2)
        assert(ex == ConstantScalarExpr(2))

    def test_OneTimesConstant2(self):
        ex = 1*ConstantScalarExpr(2)
        assert(ex == ConstantScalarExpr(2))



    # 0*constant and constant*0
    def test_ConstantTimesZero1(self):
        ex = ConstantScalarExpr(2)*ConstantScalarExpr(0)
        assert(ex == 0)

    def test_ConstantTimesZero2(self):
        ex = ConstantScalarExpr(2)*0
        assert(ex == 0)

    def test_ZeroTimesConstant1(self):
        ex = ConstantScalarExpr(0)*ConstantScalarExpr(2)
        assert(ex == 0)

    def test_ZeroTimesConstant2(self):
        ex = 1*ConstantScalarExpr(0)
        assert(ex == 0)


    # Constant divided by constant
    def test_ConstantDivideConstant(self):
        ex = ConstantScalarExpr(3.0)/ConstantScalarExpr(2.0)
        assert(ex.data()==3.0/2.0)


class TestSimplifications:

    # 0+expr and expr+0
    def test_ExprPlusZero1(self):
        y = Coordinate(1)
        ex = y + ConstantScalarExpr(0)
        assert(ex == y)

    def test_ExprPlusZero2(self):
        y = Coordinate(1)
        ex = y + 0
        assert(ex == y)

    def test_ZeroPlusExpr1(self):
        y = Coordinate(1)
        ex = 0 + y
        assert(ex == y)

    def test_ZeroPlusExpr2(self):
        y = Coordinate(1)
        ex = ConstantScalarExpr(0) + y
        assert(ex == y)

    # expr-0 and 0-expr
    def test_ExprMinusZero1(self):
        y = Coordinate(1)
        ex = y - ConstantScalarExpr(0)
        assert(ex == y)

    def test_ExprMinusZero2(self):
        y = Coordinate(1)
        ex = y - 0
        assert(ex == y)

    def test_ZeroMinusExpr1(self):
        y = Coordinate(1)
        ex = 0 - y
        assert(ex.sameas(-y))

    def test_ZeroMinusExpr2(self):
        y = Coordinate(1)
        ex = ConstantScalarExpr(0) - y
        assert(ex.sameas(-y))

    # 0*expr and expr*0
    def test_ExprTimesZero1(self):
        y = Coordinate(1)
        ex = y * ConstantScalarExpr(0)
        assert(ex == 0)

    def test_ExprTimesZero2(self):
        y = Coordinate(1)
        ex = y * 0
        assert(ex == 0)

    def test_ZeroTimesExpr1(self):
        y = Coordinate(1)
        ex = 0 * y
        assert(ex == 0)

    def test_ZeroTimesExpr2(self):
        y = Coordinate(1)
        ex = ConstantScalarExpr(0) * y
        assert(ex == 0)

    # expr*1 and 1*expr
    def test_ExprTimesOne1(self):
        y = Coordinate(1)
        ex = y * ConstantScalarExpr(1)
        assert(ex == y)

    def test_ExprTimesOne2(self):
        y = Coordinate(1)
        ex = y * 1
        assert(ex == y)

    def test_OneTimesExpr1(self):
        y = Coordinate(1)
        ex = 1 * y
        assert(ex == y)

    def test_OneTimesExpr2(self):
        y = Coordinate(1)
        ex = ConstantScalarExpr(1) * y
        assert(ex == y)

    # Expr divided by 1
    def test_ExprDividedByOne1(self):
        y = Coordinate(1)
        ex = y/ConstantScalarExpr(1)
        assert(ex == y)

    def test_ExprDividedByOne2(self):
        y = Coordinate(1)
        ex = y/1.0
        assert(ex == y)


    # Expr ** 1
    def test_ExprPowerOne1(self):
        y = Coordinate(1)
        ex = y**ConstantScalarExpr(1)
        assert(ex == y)

    def test_ExprPowerOne2(self):
        y = Coordinate(1)
        ex = y**1.0
        assert(ex == y)


    # Expr ** 0
    def test_ExprPowerZero1(self):
        y = Coordinate(1)
        ex = y**ConstantScalarExpr(0)
        assert(ex == 1)

    def test_ExprPowerZero2(self):
        y = Coordinate(1)
        ex = y**0.0
        assert(ex == 1)

    # 0 ** Expr
    def test_ZeroToAPower1(self):
        y = Coordinate(1)
        ex = ConstantScalarExpr(0)**y
        assert(ex == 0)

    def test_ZeroToAPower2(self):
        y = Coordinate(1)
        ex = y**0.0
        assert(ex == 1)


class TestArithmetic:

    def test_Add(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), 'x+y')
        assert(ex==raw)

    def test_Subtract(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), 'x-y')
        assert(ex==raw)

    def test_UnaryMinus(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), '-y')
        assert(ex==raw)

    def test_LC1(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), '3*x-2*y')
        assert(ex==raw)

    def test_LC2(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), '3*x+2*y+5')
        assert(ex==raw)

    def test_LC3(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), '5-3*x+2*y')
        assert(ex==raw)

    def test_LC4(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), 'x+4*y-(2-5*x)')
        assert(ex==raw)

    def test_Product1(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), 'x*y')
        assert(ex==raw)

    def test_Product2(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), 'x*2*y')
        assert(ex==raw)

    def test_Product3(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4), '(x+y)*(2*x-7*y+1)')
        assert(ex==raw)

    def test_Product4(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '(x+y)*(2*x-7*y+1)*(x-y)*(x*3-4*x)*(2+y)')
        assert(ex==raw)

    def test_Quotient1(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            'x/2')
        assert(ex==raw)

    def test_Quotient2(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            'x/y')
        assert(ex==raw)

    def test_Quotient3(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '2/y')
        assert(ex==raw)

    def test_Quotient4(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '(x+1)/(2+y*x)')
        assert(ex==raw)

    def test_Quotient5(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            'x/y/y/x')
        assert(ex==raw)

    def test_Power1(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            'x**2')
        assert(ex==raw)

    def test_Power2(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '2**x')
        assert(ex==raw)

    def test_Power3(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            'x**y')
        assert(ex==raw)

    def test_Power4(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '(1+x*y)**y')
        assert(ex==raw)

    def test_Power5(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '(1+x*y)**(y*2-1)')
        assert(ex==raw)



class TestComplicated:

    def test_Complicated1(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '(1+x*y)**(y*2-1)+x/y/(1+2*x + y/3 + y**x)*(1+y*x)')

    def test_Complicated2(self):
        (ex,raw) = compareEval(('x','y'), (1.5,5.4),
            '1/(1/x+1/y) - 2/(3/y+y*x**4)')

# class TestVectorsWithScalarVariables:
#
#     def test_VecTimesScalar1(self):
#         (ex,raw) = compareEval(('x','y'), (1.5,5.4),
#             'x*np.array([1,2,3])')



class TestExpectedErrors:

    def test_ZeroDiv1(self):
        with pytest.raises(ZeroDivisionError) as err_info:
            ex = evalExpr(('x','y'), (1.5,5.4),'x/0')
        print('detected expected exception')
        assert('by zero' in str(err_info.value))

    def test_ZeroDiv2(self):
        with pytest.raises(ZeroDivisionError) as err_info:
            ex = evalExpr(('x','y'), (1.5,5.4),'2*y + x/0')
        print('detected expected exception')
        assert('by zero' in str(err_info.value))
