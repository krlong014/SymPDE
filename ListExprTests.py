import pytest
import numpy as np
from Expr import (Expr, ConstantScalarExpr, ConstantVectorExpr,
    Coordinate, ListExpr)
from SimpleEvaluator import compareEval, evalExpr



class TestLists:

    def test_LengthOneList(self):
        x = Coordinate(0)
        L = ListExpr(x)
        assert(L[0]==x)

    def test_ListFromVArgs(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L = ListExpr(x, y)
        assert(L[0]==x and L[1]==y)


    def test_ListFromTuple(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L = ListExpr((x,y))
        assert(L[0]==x and L[1]==y)


    def test_ListFromList(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L = ListExpr([x,y])
        assert(L[0]==x and L[1]==y)


    def test_ListByAppending(self):
        x = Coordinate(0)
        y = Coordinate(1)
        z = Coordinate(2)
        L = ListExpr(x)
        L.append(y)
        L.append(z)
        assert(L[0]==x and L[1]==y and L[2]==z)

    def test_ListFromNumber1(self):
        L = ListExpr(1)
        assert(L[0].sameas(ConstantScalarExpr(1)))

    def test_ListFromNumber2(self):
        x = Coordinate(0)
        L = ListExpr(1, x)
        assert(L[0].sameas(ConstantScalarExpr(1)) and L[1]==x)

    def test_ListFromNumpy(self):
        x = Coordinate(0)
        a = np.array([1,2,3])
        L = ListExpr(1, x, a)
        print(L)
        assert(L[0].sameas(ConstantScalarExpr(1))
            and L[1]==x
            and L[2].sameas(ConstantVectorExpr(a)))

    def test_ListIter(self):
        x = Coordinate(0)
        a = np.array([1,2,3])
        a2 = Expr._convertToExpr(a)
        c = 1
        c2 = Expr._convertToExpr(c)
        L = ListExpr(c, x, a)
        tup = (c2, x, a2)
        same = True
        for e1,e2 in zip(L, tup):
            same = same and (e1.sameas(e2))

        assert(same)

    def test_ListComparison1(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = ListExpr(x, y)
        L2 = ListExpr(x, y)
        assert(L1.sameas(L2))

    def test_ListComparison2(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = ListExpr(x, y)
        L2 = ListExpr(y, x)
        assert(not L1.sameas(L2))

    def test_ListComparison3(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = ListExpr(x, y)
        L2 = ListExpr(x)
        assert(not L1.sameas(L2) and not L2.sameas(L1))

    def test_ListComparison4(self):
        x = Coordinate(0)
        y = Coordinate(1)
        L1 = ListExpr(x, y)
        E2 = x + y
        assert(not L1.sameas(E2) and not E2.sameas(L1))





class TestExpectedListErrors:

    def test_InvalidListInputNotAnExpr1(self):
        with pytest.raises(ValueError) as err_info:
            L = ListExpr('not an Expr')

        print('detected expected exception: {}'.format(err_info))
        assert('not convertible' in str(err_info.value))


    def test_InvalidListInputNotAnExpr2(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L = ListExpr((x, 'not an Expr'))

        print('detected expected exception: {}'.format(err_info))
        assert('not convertible' in str(err_info.value))

    def test_InvalidListInputListWithinList(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = ListExpr(x)
            L2 = ListExpr(y, L1)


        print('detected expected exception: {}'.format(err_info))
        assert('List within list' in str(err_info.value))

    def test_NoAddingLists1(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = ListExpr(x)
            f = y + L1


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoAddingLists2(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = ListExpr(x)
            L2 = ListExpr(y)
            f = L1 + L2


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoMultiplyingLists1(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = ListExpr(x)
            f = y * L1


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoMultiplyingLists2(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            L1 = ListExpr(x)
            L2 = ListExpr(y)
            f = L1 * L2


        print('detected expected exception: {}'.format(err_info))
        assert('not compatible' in str(err_info.value))

    def test_NoNegatingLists(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L1 = ListExpr(x)
            f = -L1


        print('detected expected exception: {}'.format(err_info))
        assert('cannot negate' in str(err_info.value))

    def test_NoDividingByLists(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L1 = ListExpr(x)
            f = 2/L1


        print('detected expected exception: {}'.format(err_info))
        assert('Division by list' in str(err_info.value))

    def test_NoDividingOfLists(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            L1 = ListExpr(x)
            f = L1/2


        print('detected expected exception: {}'.format(err_info))
        assert('Dividing a list' in str(err_info.value))
