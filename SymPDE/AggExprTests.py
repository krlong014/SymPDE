import pytest
import numpy as np
from Expr import (Expr, ConstantScalarExpr, ConstantVectorExpr,
    Coordinate, AggExpr)
from SimpleEvaluator import compareEval, evalExpr



class TestAggs:

    def test_LengthOneAgg(self):
        x = Coordinate(0)
        L = AggExpr(x)
        assert(L[0]==x)

    def test_AggFromVArgs(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L = AggExpr(x, y)
        assert(L[0]==x and L[1]==y)


    def test_AggFromTuple(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L = AggExpr((x,y))
        assert(L[0]==x and L[1]==y)


    def test_AggFromAgg(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L = AggExpr([x,y])
        assert(L[0]==x and L[1]==y)


    def test_AggByAppending(self):
        x = Coordinate(0)
        y = Coordinate(1)
        z = Coordinate(2)
        L = AggExpr(x)
        L.append(y)
        L.append(z)
        assert(L[0]==x and L[1]==y and L[2]==z)

    def test_AggFromNumber1(self):
        L = AggExpr(1)
        assert(L[0].sameas(ConstantScalarExpr(1)))

    def test_AggFromNumber2(self):
        x = Coordinate(0)
        L = AggExpr(1, x)
        assert(L[0].sameas(ConstantScalarExpr(1)) and L[1]==x)

    def test_AggFromNumpy(self):
        x = Coordinate(0)
        a = np.array([1,2,3])
        L = AggExpr(1, x, a)
        print(L)
        assert(L[0].sameas(ConstantScalarExpr(1))
            and L[1]==x
            and L[2].sameas(ConstantVectorExpr(a)))

    def test_AggIter(self):
        x = Coordinate(0)
        a = np.array([1,2,3])
        a2 = Expr._convertToExpr(a)
        c = 1
        c2 = Expr._convertToExpr(c)
        L = AggExpr(c, x, a)
        tup = (c2, x, a2)
        same = True
        for e1,e2 in zip(L, tup):
            same = same and (e1.sameas(e2))

        assert(same)

    def test_AggComparison1(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = AggExpr(x, y)
        L2 = AggExpr(x, y)
        assert(L1.sameas(L2))

    def test_AggComparison2(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = AggExpr(x, y)
        L2 = AggExpr(y, x)
        assert(not L1.sameas(L2))

    def test_AggComparison3(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = AggExpr(x, y)
        L2 = AggExpr(x)
        assert(not L1.sameas(L2) and not L2.sameas(L1))

    def test_AggComparison4(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = AggExpr(x, y)
        E2 = x + y
        assert(not L1.sameas(E2) and not E2.sameas(L1))





class TestExpectedAggErrors:

    def test_InvalidAggInputNotAnExpr1(self):
        with pytest.raises(ValueError) as err_info:
            L = AggExpr('not an Expr')

        print('detected expected exception: {}'.format(err_info))
        assert('not convertible' in str(err_info.value))


    def test_InvalidAggInputNotAnExpr2(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L = AggExpr((x, 'not an Expr'))

        print('detected expected exception: {}'.format(err_info))
        assert('not convertible' in str(err_info.value))

    def test_InvalidAggInputAggWithinAgg(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = AggExpr(x)
            L2 = AggExpr(y, L1)


        print('detected expected exception: {}'.format(err_info))
        assert('Agg within list' in str(err_info.value))

    def test_NoAddingAggs1(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = AggExpr(x)
            f = y + L1


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoAddingAggs2(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = AggExpr(x)
            L2 = AggExpr(y)
            f = L1 + L2


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoMultiplyingAggs1(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = AggExpr(x)
            f = y * L1


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoMultiplyingAggs2(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = AggExpr(x)
            L2 = AggExpr(y)
            f = L1 * L2


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoNegatingAggs(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L1 = AggExpr(x)
            f = -L1


        print('detected expected exception: {}'.format(err_info))
        assert('cannot negate' in str(err_info.value))

    def test_NoDividingByAggs(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L1 = AggExpr(x)
            f = 2/L1


        print('detected expected exception: {}'.format(err_info))
        assert('Division by list' in str(err_info.value))

    def test_NoDividingOfAggs(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L1 = AggExpr(x)
            f = L1/2


        print('detected expected exception: {}'.format(err_info))
        assert('Dividing a list' in str(err_info.value))
