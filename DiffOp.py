from abc import ABC, abstractmethod
from Expr import Expr, UnaryExpr, Coordinate, ListExpr
from ExprShape import (ExprShape, ScalarShape, TensorShape, VectorShape)
from VectorExprs import Vector
from SymbolicFunction import SymbolicFunctionBase
import pytest

class DiffOp(UnaryExpr):

    def __init__(self, op, arg):
        assert(isinstance(op, HungryDiffOp))
        if op.acceptShape(arg.shape()):
            myShape = op.outputShape(arg.shape())
        else:
            raise ValueError('Undefined DiffOp action: {} acting on {}'.format(op, arg))

        super().__init__(arg, myShape)
        self.op = op


    def __str__(self):
        return '{}({})'.format(self.op.__str__(), self.arg.__str__())



class DiffOpOnFunction(DiffOp):

    def __init__(self, op, arg):
        assert(isinstance(arg, SymbolicFunctionBase))
        super().init(op, arg)

    def funcID(self):
        return self.


class HungryDiffOp(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def acceptShape(self, input):
        pass

    @abstractmethod
    def outputShape(self, input):
        pass


    def __call__(self, arg):

        # Make sure the type makes sense
        if not Expr._convertibleToExpr(arg):
            raise TypeError(
                'diff op [{}] cannot accept type [{}]'.format(self, arg)
                )

        f = Expr._convertToExpr(arg)

        # Make sure the operator and input are consistent
        if not self.acceptShape(f.shape()):
            raise TypeError(
                'diff op [{}] cannot accept argument [{}]'.format(self,f)
                )


        # Form the diff op expression
        if isinstance(f, SymbolicFunctionBase):
            return DiffOpOnFunction(self, f)
        return DiffOp(self, f)


class _IdentityOp(HungryDiffOp):
    def __init__(self):
        super().__init__()

    def __call__(self, arg):
        return arg

    def acceptShape(self, input):
        return True

    def outputShape(self, input):
        return input

    def __str__(self):
        return 'IdentityOp'

def Partial(f, coord):
    if isinstance(coord, int):
        op = _Partial(coord)
    elif isinstance(coord, Coordinate):
        op = _Partial(coord.dir)
    else:
        raise ValueError('unable to interpret direction {} in Partial()'.format(coord))

    return op(f)

class _Partial(HungryDiffOp):
    def __init__(self, dir, name=None):
        super().__init__()
        self.dir = dir
        if name==None:
            self.myName = Expr._dirName(dir)
        else:
            self.myName = name

    def acceptShape(self, input):
        return True

    def outputShape(self, input):
        return input

    def __str__(self):
        return 'd_d{}'.format(self.myName)

def Partial(f, coord):
    if isinstance(coord, int):
        op = _Partial(coord)
    elif isinstance(coord, Coordinate):
        op = _Partial(coord.dir)
    else:
        raise ValueError('unable to interpret direction {} in Partial()'.format(coord))

    return op(f)


class _Div(HungryDiffOp):

    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return not isinstance(input, ScalarShape)

    def outputShape(self, input):
        if isinstance(input, (VectorShape, VectorShape)):
            return ScalarShape()
        if isinstance(input, TensorShape):
            return VectorShape(input.dim)
        raise ValueError(
            'Div.outputShape expected vector or tensor, got [{}]'.format(input)
            )
    def __str__(self):
        return 'Div'


def Div(f):
    div = _Div()
    return div(f)


class _Gradient(HungryDiffOp):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim

    def acceptShape(self, input):
        return not isinstance(input, TensorShape)

    def outputShape(self, input):
        if isinstance(input, ScalarShape):
            return VectorShape(self.dim)
        if isinstance(input, VectorShape):
            return TensorShape(self.dim)
        raise ValueError('grad.outputShape expected scalar or vector, got [{}]'.format(input))

    def __str__(self):
        return 'Grad'

class _Curl(HungryDiffOp):
    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return isinstance(input, VectorShape) and input.dim==3

    def outputShape(self, input):
        assert(self.acceptShape(input))
        return input

    def __str__(self):
        return 'Curl'


def Curl(f):
    curl = _Curl()
    return curl(f)


class _Rot(HungryDiffOp):
    def __init__(self):
        super().__init__()

    def acceptShape(self, input):
        return isinstance(input, VectorShape) and input.dim==2

    def outputShape(self, input):
        assert(self.acceptShape(input))
        return ScalarShape()

    def __str__(self):
        return 'Rot'



def Rot(f):
    rot = _Rot()
    return rot(f)



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

        print('Div(F)=', divF)
        assert(divF.sameas(DiffOp(div, F)) and divF.shape()==ScalarShape())


    def test_Curl(self):

        x = Coordinate(0)
        y = Coordinate(1)
        z = Coordinate(2)

        curl = _Curl()
        F = Vector(x,y,z)
        curlF = Curl(F)

        print('Curl(F)=', curlF)
        assert(curlF.sameas(DiffOp(curl, F)) and curlF.shape().dim==3)


    def test_Rot(self):

        x = Coordinate(0)
        y = Coordinate(1)

        rot = _Rot()
        F = Vector(x,y)
        rotF = rot(F)

        print('Rot(F)=', rotF)
        assert(rotF.sameas(DiffOp(rot, F)) and rotF.shape()==ScalarShape())



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



    def test_DiffOpOfList(self):
        with pytest.raises(TypeError) as err_info:
            x = Coordinate(0)
            y = Coordinate(1)
            z = Coordinate(2)
            L = ListExpr(x,y,z)

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
