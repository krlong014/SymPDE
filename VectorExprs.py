from Expr import (Expr, Coordinate, VectorExprInterface, VectorExprIterator,
                    AggExpr, ConstantVectorExpr)
from ExprShape import (ExprShape, ScalarShape, TensorShape,
        VectorShape, AggShape)
from numpy import ndarray, array
from numbers import Number
import pytest

def Vector(*args):
    '''Create a Vector expression.'''
    # if the input is a numpy array, put it into a ConstantVectorExpr

    if len(args)==1:
        x = args[0]
    else:
        x = args

    if isinstance(x, ndarray):
        order = len(x.shape)
        if order != 1:
            raise ValueError('Non-vector input [{}] to Vector()'.format(x))
        if len(x) <= 1:
            raise ValueError('1D input [{}] to Vector()'.format(x))
        return ConstantVector(x)

    # if the input is a 1D list or tuple:
    #   (*) Form a ConstantVectorExpr if all the elements are constants
    #   (*) Form a VectorExpr otherwise
    if isinstance(x, (list, tuple, AggExpr)):
        allConsts = True
        elems = []
        if len(x) == 1:
            raise ValueError('Impossible to convert a 1D object to a vector')

        for x_i in x:

            if isinstance(x_i, Number):
                elems.append(x_i)
            elif isinstance(x_i, Expr):
                if not isinstance(x_i.shape(), ScalarShape):
                    raise ValueError('Vector() element not scalar: [{}]'.format(x_i))
                if not x_i.isConstant():
                    allConsts = False
                    elems.append(x_i)
                else:
                    elems.append(Expr._convertToExpr(x_i.data()))
            else:
                raise ValueError('Vector() input neither number nor expr: [{}]'.format(x_i))

        if allConsts:
            return ConstantVectorExpr(array(elems))
        else:
            exprElems = []
            for e in elems:
                exprElems.append(Expr._convertToExpr(e))
            return AggedVectorExpr(exprElems)

    # If the input is a tensor, this won't work
    if isinstance(x, Expr) and isinstance(x.shape(), TensorShape):
        raise ValueError('impossible to convert a tensor to a vector')

    # If the input is a scalar, this won't work
    if isinstance(x, Expr) and isinstance(x.shape(), ScalarShape):
        raise ValueError('pointless to convert a scalar to a vector')

    # Don't know what this input is
    raise ValueError('bad input {} to Vector()')


class AggedVectorExpr(Expr, VectorExprInterface):

    def __init__(self, elems):
        self._elems = elems
        super().__init__(VectorShape(len(elems)))

    def __getitem__(self, i):
        return self._elems[i]

    def __len__(self):
        return len(self._elems)

    def __iter__(self):
        return VectorExprIterator(self)

    def __str__(self):
        rtn = 'Vector('
        for i,e in enumerate(self._elems):
            if i>0:
                rtn += ', '
            rtn += e.__str__()
        rtn += ')'
        return rtn

    def _sameas(self, other):
        if len(self)!=len(other):
            return False

        for (me, you) in zip(self, other):
            if not me.sameas(you):
                return False
        return True







class TestVectorSanity:

    def test_VecGetElem1(self):
        x = Coordinate(0)
        y = Coordinate(1)
        v = Vector(x, y)
        assert(v[0]==x and v[1]==y and len(v)==2)

    def test_VecSameas(self):
        x = Coordinate(0)
        y = Coordinate(1)
        v = Vector(x, y)
        u = Vector(x, y)
        assert(v.sameas(u))

    def test_ConstantVec1(self):
        x = 1
        y = 2
        v = Vector(x, y)

        assert(isinstance(v, ConstantVectorExpr))

    def test_ConstantVec2(self):
        x = 1
        y = 2
        v = 3.14*Vector(x, y)

        assert(isinstance(v, ConstantVectorExpr))

    def test_ConstantVec3(self):
        x = 1
        y = 2
        v = Vector(x, y)*3.14

        assert(isinstance(v, ConstantVectorExpr))




class TestDiffOpExpectedErrors:

    def test_VectorNonsenseInput(self):
        with pytest.raises(ValueError) as err_info:
            x = 'not expr'
            bad = Vector(x)


        print('detected expected exception: {}'.format(err_info))
        assert('bad input' in str(err_info.value))


    def test_VectorScalarInput1(self):
        with pytest.raises(ValueError) as err_info:
            x = Coordinate(0)
            bad = Vector(x)


        print('detected expected exception: {}'.format(err_info))
        assert('pointless to convert' in str(err_info.value))

    def test_VectorScalarInput2(self):
        with pytest.raises(ValueError) as err_info:
            bad = Vector(array([1.0]))


        print('detected expected exception: {}'.format(err_info))
        assert('1D input' in str(err_info.value))

    def test_VectorScalarInput3(self):
        with pytest.raises(ValueError) as err_info:
            bad = Vector(1.0, 'not expr')


        print('detected expected exception: {}'.format(err_info))
        assert('input neither number' in str(err_info.value))

    def test_VectorScalarInput4(self):
        with pytest.raises(ValueError) as err_info:
            class Blah:
                def __init__(self):
                    pass
            bad = Vector(1.0, Blah())


        print('detected expected exception: {}'.format(err_info))
        assert('neither number nor expr' in str(err_info.value))
