from abc import ABC, abstractmethod
from ExprShape import (ExprShape, ScalarShape, TensorShape,
    VectorShape, ListShape)
from numbers import Number
from numpy.linalg import norm
from numpy import dot, array_equiv, inf, ndarray
import copy

class Expr(ABC):
    """
    Expr is the base class for symbolic expressions
    """
    def __init__(self, shape):
        """
        Constructor for base class
        """
        assert(isinstance(shape, ExprShape))
        self._shape = shape

    def shape(self):
        return self._shape

    def __len__(self):
        return self.shape().dim

    def sameas(self, other):
        return (type(self)==type(other)
            and self.shape().sameas(other.shape())
            and self._sameas(other))

    @abstractmethod
    def _sameas(self, other):
        pass

    def isConstant(self):
        return False

    def isSpatialConstant(self):
        return False

    def getConstantValue(self):
        return None

    def __neg__(self):
        """Unary minus operator"""
        if isinstance(self, ConstantExprBase):
            rtn = copy.deepcopy(self)
            rtn._data = -rtn._data
            return rtn
        if isinstance(self, ListExpr):
            raise ValueError('cannot negate a ListExpr')
        return UnaryMinus(self)

    def __add__(self, other):
        """Add two expressions"""
        return Expr._addOrSubtract(self, other, 1)

    def __sub__(self, other):
        """Subtract two expressions"""
        return Expr._addOrSubtract(self, other, -1)


    def __mul__(self, other):
        """Multiply two expressions"""
        return Expr._multiply(self, other)

    def __truediv__(self, other):
        """Divide two expressions"""
        return Expr._divide(self, other)

    def __pow__(self, other):
        """Raise an expression to a power"""
        return Expr._power(self, other)

    def __radd__(self, other):
        """Add an expression to a constant"""
        return Expr._addOrSubtract(other, self, 1)

    def __rsub__(self, other):
        """Subtract an expression from a constant"""
        return Expr._addOrSubtract(other, self, -1)


    def __rmul__(self, other):
        """Multiply a constant by an expression"""
        return Expr._multiply(other, self)

    def __rtruediv__(self, other):
        """Divide a constant by an expression"""
        return Expr._divide(other, self)

    def __rpow__(self, other):
        """Raise a constant to the power of an expression"""
        return Expr._power(other, self)


    ## -------------- internal Expr methods ----------------

    def _addOrSubtract(leftIn, rightIn, sign):

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)

        # Check for compatibility
        ExprShape.assertAdditiveCompatibility(Expr._getShape(left),
            Expr._getShape(right))

        # Adding zero does nothing
        if Expr._isZero(left):
            if sign<0: return -right
            return right
        if Expr._isZero(right):
            return left

        # ----- Catch constant argument cases
        if left.isConstant() and right.isConstant():
            return left.plusConstant(right, sign)

        # ----- Default case: expr plus/minus expr
        return SumExpr(left, right, sign)


    def _multiply(leftIn, rightIn):

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)

        # Check for compatibility
        ExprShape.assertMultiplicativeCompatibility(Expr._getShape(left),
            Expr._getShape(right))

        # Zero times anything is zero
        if Expr._isZero(left) or Expr._isZero(right):
            return 0

        # Multiplying by the identity does nothing
        if Expr._isIdentity(right):
            return left
        if Expr._isIdentity(left):
            return right

        # If both are constant, multiply now
        if left.isConstant() and right.isConstant():
            return left.timesConstant(right)

        # Expr times expr
        if (isinstance(left.shape(), ScalarShape)
                or isinstance(right.shape(), ScalarShape)):
            return ProductExpr(left, right)

        return DotProductExpr(left, right)


    def _divide(leftIn, rightIn):

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)


        # Division by a list is an error
        if isinstance(right, ListExpr):
            raise ValueError(
                'Division by list is undefined: num={}, den={}'.format(left,right))

        # Division of a list is an error
        if isinstance(left, ListExpr):
            raise ValueError(
                'Dividing a list is undefined: num={}, den={}'.format(left,right))

        # Division by a vector or tensor is an error
        if Expr._getShape(right) != ScalarShape():
            raise ValueError(
                'Division of [{}] by non-scalar [{}] is undefined.'
                    .format(left.__repr__(), right.__repr__())
                )


        # Division by zero is an error
        if Expr._isZero(right):
            raise ZeroDivisionError(
                'Dividing [{}] by zero'.format(right)
                )

        # Division by one does nothing
        if Expr._isIdentity(right):
            return left

        # Check for constant divided by constant
        if left.isConstant() and right.isConstant():
            rtn = copy.deepcopy(left)
            rtn._data = rtn._data/right._data
            return rtn

        # Convert division by a constant to multiplication by its reciprocal
        if right.isConstant():
            return ProductExpr(left, ConstantScalarExpr(1.0/right._data))

        # Default case: Expr/Expr
        return QuotientExpr(left, right)


    def _power(baseIn, powerIn):

        # Convert input to Expr
        base = Expr._convertToExpr(baseIn)
        power = Expr._convertToExpr(powerIn)

        # Exponentiation with non-scalars is undefined here
        if Expr._getShape(base) != ScalarShape():
            raise ValueError('Non-scalar base [{}] in Expr._power'.format(base))

        if Expr._getShape(power) != ScalarShape():
            raise ValueError('Non-scalar power [{}] in Expr._power'.format(power))

        # Check for zero**zero
        if Expr._isZero(base) and Expr._isZero(power):
            raise ValueError('Expr._power detected 0**0')

        # Zero to any nonzero power is zero
        if Expr._isZero(base):
            return 0

        # Any nonzero base to the zero power is 1
        if Expr._isZero(power):
            return 1

        # Any nonzero base to the power 1 is that base
        if not Expr._isZero(base) and Expr._isIdentity(power):
            return base

        # Evaluate constant**constant directly
        if base.isConstant() and power.isConstant():
            return ConstantScalarExpr((base._data)**(power._data))

        # General case
        return PowerExpr(base, power)



    def _getShape(x):
        if isinstance(x, Expr):
            return x.shape()
        elif isinstance(x, Number):
            return ScalarShape()


    def _isConstant(x):
        return ((isinstance(x, Expr) and x.isConstant())
                or isinstance(x, Number))

    def _isZero(x):
        if not x.isConstant():
            return False
        return x._isZero()
        raise ValueError('Expr._isZero got bad arg [{}]'.format(x))

    def _isIdentity(x):
        if not x.isConstant():
            return False
        if isinstance(x, ConstantScalarExpr):
            return x.data()==1.0
        # if isinstance(x, ConstantTensorExpr):
        #     d = x.data()
        #     for i in range(x.shape().dim):
        #         for j in range(x.shape().dim):
        #             if ((i!=j and d[i,j] != 0.0) or (i==j and d[i,j] != 1.0)):
        #                 return False
        #     return True
        return False


    def _maybeParenthesize(x):
        if isinstance(x, SumExpr):
            return '({})'.format(x)
        return '{}'.format(x)


    def _dirName(dir):
        cart = ('x', 'y', 'z')
        return cart[dir]


    def _convertToExpr(x):
        if isinstance(x, Expr):
            return x

        if isinstance(x, Number):
            return ConstantScalarExpr(x)

        if isinstance(x, ndarray):
            return ConstantVectorExpr(x)

        raise ValueError('input [{}] cannot be converted to Expr'.format(x))

    def _convertibleToExpr(x):
        return isinstance(x, (Expr, Number, ndarray))





#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################

class ExprWithChildren(Expr):
    def __init__(self, children, shape):

        if not (isinstance(children, (list, tuple)) and len(children)>0):
            raise ValueError('ExprWithChildren bad input {}'.format(children))
        for c in children:
            assert(not isinstance(c, ListExpr))

        super().__init__(shape)

        self._children = children

    def children(self):
        return self._children

    def child(self, i):
        return self._children[i]

    def _sameas(self, other):
        for me, you in zip(self._children, other._children):
            if not me._sameas(you):
                return False
        return True

    def isSpatialConstant(self):
        for c in self._children:
            if not c.isSpatialConstant():
                return False
        return True

    def isConstant(self):
        for c in self._children:
            if not c.isConstant():
                return False
        return True

class UnaryExpr(ExprWithChildren):
    def __init__(self, arg, shape):
        super().__init__((arg,),shape)

    def arg(self):
        return self.child(0)

    def _sameas(self, other):
        return self.arg().sameas(other.arg())



class UnaryMinus(UnaryExpr):
    def __init__(self, arg):
        super().__init__(arg, Expr._getShape(arg))

    def __str__(self):
        return '-%s' % Expr._maybeParenthesize(self.arg())

    def __repr__(self):
        return 'UnaryMinus[arg={}]'.format(self.arg().__repr__())


class BinaryExpr(ExprWithChildren):
    def __init__(self, L, R, shape):
        super().__init__((L, R), shape)

    def left(self):
        return self.child(0)

    def right(self):
        return self.child(1)

    def toString(self, op):
        return '{}{}{}'.format(Expr._maybeParenthesize(self.left()),
            op, Expr._maybeParenthesize(self.right()))

    def isSpatialConstant(self):
        return self.left().isSpatialConstant() and self.right().isSpatialConstant()

class SumExpr(BinaryExpr):
    def __init__(self, L, R, sign):
        super().__init__(L, R, Expr._getShape(L))
        self.sign = sign
        if sign==1:
            self.op=' + '
        elif sign==-1:
            self.op=' - '
        else:
            raise ValueError('SumExpr: Nonsense sign argument {}'.format(sign))

    def _sameas(self, other):
        return (self.left().sameas(other.left()) and self.right().sameas(other.right())
            and self.sign==other.sign)

    def __str__(self):
        return super().toString(self.op)

    def __repr__(self):
        return 'SumExpr[left={}, right={}, shape={}]'.format(self.left().__repr__(),
            self.right().__repr__(), self.shape())


class ProductExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())

    def __str__(self):
        return super().toString('*')

    def __repr__(self):
        return 'ProductExpr[left={}, right={}]'.format(self.left().__repr__(),
            self.right().__repr__())


def Dot(a, b):
    assert(isinstance(a.shape(), VectorShape)
        and isinstance(b.shape(), VectorShape))
    assert(a.shape().dim == b.shape().dim)

    return DotProductExpr(a,b)


class DotProductExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ExprShape.productShape(L.shape(), R.shape()))

    def __str__(self):
        return 'dot({},{})'.format(self.left(), self.right())

    def __repr__(self):
        return 'DotProductExpr[left={}, right={}, shape={}]'.format(self.left().__repr__(),
            self.right().__repr__(), self.shape())

def Cross(a, b):
    assert(isinstance(a.shape(), VectorShape)
        and isinstance(b.shape(), VectorShape))
    assert(a.shape().dim == b.shape().dim)

    return CrossProductExpr(a,b,a.shape())

class CrossProductExpr(BinaryExpr):
    def __init__(self, L, R, shape):
        pass

    def __str__(self):
        return 'cross({},{})'.format(self.L, self.R)

class QuotientExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, L.shape())

    def __str__(self):
        return super().toString('/')



class PowerExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())

    def __str__(self):
        return super().toString('**')



#############################################################################
#
# Base class for constant expressions
#
#############################################################################


class ConstantExprBase(Expr, ABC):
    def __init__(self, data, shape):
        super().__init__(shape)
        self._data = data

    def isConstant(self):
        return True

    def isSpatialConstant(self):
        return True

    def data(self):
        return self._data

    def plusConstant(self, other, sign):
        assert(isinstance(other, ConstantExprBase))
        assert(type(self)==type(other))

        return type(self)(self._data + sign*other._data)

    @abstractmethod
    def timesConstant(self, other):
        pass

    def __str__(self):
        return '{}'.format(self._data)


    def __repr__(self):
        return '{}[data={}, shape={}]'.format(self.typename(),
            self._data, self.shape())

    def __eq__(self, other):
        return self.sameas(other)

    def _sameas(self, other):
        return array_equiv(self._data, other._data)

    def _isZero(self):
        return norm(self._data, ord=inf)==0.0



class ConstantScalarExpr(ConstantExprBase):
    def __init__(self, data):
        super().__init__(data, ScalarShape())

    def typename(self):
        return "ConstantScalar"

    def timesConstant(self, other):
        assert(isinstance(other, ConstantExprBase))

        # We can deal with scalar*scalar here
        if isinstance(other, ConstantScalarExpr):
            return ConstantScalarExpr(self.data() * other.data())

        # Scalar multiplication is always commutative, so we reverse the order
        # to delegate the operation to the other type.
        return other.timesConstant(self)

    def _isZero(self):
        return self.data()==0.0


class VectorExprInterface(ABC):

    @abstractmethod
    def __getitem__(self, i):
        pass



class VectorExprIterator:
    '''Iterator for elements of vectors'''
    def __init__(self, parent):
        '''Constructor'''
        self._index = 0
        assert(isinstance(parent, VectorExprInterface))
        self._parent = parent

    def __next__(self):
        '''Advance the iterator'''
        if self._index>=0 and self._index < len(self._parent):
            result = self._parent._elems[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration


class VectorElementInterface:

    def __init__(self, parent, index):
        self._parent = parent
        self._index = index

    def parent(self):
        return _parent

    def index(self):
        return _index

    def __str__(self):
        return '{}[{}]'.format(self.parent(), self.index())




class ConstantVectorExpr(ConstantExprBase, VectorExprInterface):
    def __init__(self, data):
        shape = VectorShape(len(data))
        super().__init__(data, shape)


    def typename(self):
        return "ConstantVector"

    def timesConstant(self, other):
        assert(isinstance(other, ConstantExprBase))

        resultShape = ExprShape.productShape(self.shape(), other.shape())

        if isinstance(resultShape, ScalarShape):
            return ConstantScalarExpr(self.data() * other.data())
        if isinstance(resultShape, VectorShape):
            return ConstantVectorExpr(dot(self.data(), other.data()))

    def __getitem__(self, i):
        assert(i>=0 and i<self.shape().dim)
        return self.data()[i]

    def __len__(self):
        return self.shape().dim





#############################################################################
#
# Class for coordinate functions (e.g., x, y, z)
#
#############################################################################


class Coordinate(Expr):

    def __init__(self, dir, name=None):
        super().__init__(ScalarShape())
        self.dir = dir
        if name==None:
            self.name = Expr._dirName(dir)
        else:
            self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Coordinate[dir={}, name={}, shape={}]'.format(self.dir,
            self.name, self.shape())

    def _sameas(self, other):
        return self.dir==other.dir and self.name==other.name


#############################################################################
#
# Class for listing expressions
#
#############################################################################

class ListExpr(Expr):

    def __init__(self, *args):
        super().__init__(ListShape())

        if len(args)==1:
            input = args[0]
        else:
            input = args


        if not isinstance(input, (list, tuple, Expr, Number, ndarray)):
            raise ValueError('input [{}] not convertible to ListExpr'.format(input))

        if isinstance(input, ListExpr):
            self.data = input.data
        elif isinstance(input, (list, tuple)):
            self.data = []
            for i,e in enumerate(input):
                if isinstance(e, ListExpr):
                    raise ValueError('List within list detected in entry \
                    #{}=[]'.format(i, e))
                if not Expr._convertibleToExpr(e):
                    raise ValueError('List entry #{}=[] not convertible \
                    to Expr'.format(i,e))
                self.data.append(Expr._convertToExpr(e))
        else:
            self.data = [Expr._convertToExpr(input),]

    def _sameas(self, other):
        if len(self)!=len(other):
            return False

        for (me, you) in zip(self, other):
            if not me.sameas(you):
                return False
        return True



    def __getitem__(self, i):
        if i<0 or i>=(len(self)):
            raise(IndexError('Index {} out of range [0,{}]'.format(i, len(self)-1)))
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __contains__(self, x):
        return x in self.data

    def __iter__(self):
        return ListExprIterator(self)

    def __str__(self):
        rtn = 'List('
        for i,e in enumerate(self.data):
            if i>0:
                rtn += ', '
            rtn += e.__str__()
        rtn += ')'
        return rtn

    def append(self, entry):
        self.data.append(entry)



class ListExprIterator:
    '''Iterator for expressions stored in containers'''
    def __init__(self, parent):
        '''Constructor'''
        self._index = 0
        assert(isinstance(parent, ListExpr))
        self._parent = parent

    def __next__(self):
        '''Advance the iterator'''
        if self._index>=0 and self._index < len(self._parent):
            result = self._parent.data[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
