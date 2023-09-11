from SymPDE import Vector, Coordinate
import SymPDE
import numpy as np
import pytest

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

        assert(v.isConstant())

    def test_ConstantVec2(self):
        x = 1
        y = 2
        v = 3.14*Vector(x, y)

        assert(v.isConstant())

    def test_ConstantVec3(self):
        x = 1
        y = 2
        v = Vector(x, y)*3.14

        assert(v.isConstant())




class TestVectorExpectedErrors:

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
            bad = Vector(np.array([1.0]))


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
