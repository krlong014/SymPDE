from abc import ABC, abstractmethod
from ExprShape import ExprShape, ScalarShape, TensorShape, VectorShape
from numbers import Number
from numpy.linalg import norm
from numpy import dot, array_equiv, inf
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

    def sameas(self, other):
        return (type(self)==type(other)
            and self.shape().sameas(other.shape())
            and self._sameas(other))

    def _sameas(self, other):
        return False

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

        # Default case: expr times expr
        return ProductExpr(left, right)


    def _divide(leftIn, rightIn):

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)

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
            print('left=', rtn.__repr__())
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
        if isinstance(x, ConstantTensorExpr):
            d = x.data()
            for i in range(x.shape().dim):
                for j in range(x.shape().dim):
                    if ((i!=j and d[i,j] != 0.0) or (i==j and d[i,j] != 1.0)):
                        return False
            return True
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

        raise ValueError('input [{}] cannot be converted to Expr'.format(x))






#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################


class UnaryExpr(Expr):
    def __init__(self, arg, shape):
        super().__init__(shape)
        self.arg = arg


class UnaryMinus(UnaryExpr):
    def __init__(self, arg):
        super().__init__(arg, Expr._getShape(arg))

    def _sameas(self, other):
        return self.arg.sameas(other.arg)

    def __str__(self):
        return '-%s' % Expr._maybeParenthesize(self.arg)

    def __repr__(self):
        return 'UnaryMinus[arg={}]'.format(self.arg.__repr__())


class BinaryExpr(Expr):
    def __init__(self, L, R, shape):
        super().__init__(shape)
        self.L = L
        self.R = R

    def _sameas(self, other):
        return (self.L.sameas(other.L) and self.R.sameas(other.R))

    def toString(self, op):
        return '{}{}{}'.format(Expr._maybeParenthesize(self.L),
            op, Expr._maybeParenthesize(self.R))



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
        return (self.L.sameas(other.L)
            and self.R.sameas(other.R) and self.sign==other.sign)

    def __str__(self):
        return super().toString(self.op)

    def __repr__(self):
        return 'SumExpr[left={}, right={}, shape={}]'.format(self.L.__repr__(),
            self.R.__repr__(), self.shape())


class ProductExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ExprShape.productShape(L.shape(), R.shape()))

    def __str__(self):
        return super().toString('*')

    def __repr__(self):
        return 'ProductExpr[left={}, right={}, shape={}]'.format(self.L.__repr__(),
            self.R.__repr__(), self.shape())


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
        return '%g' % self._data


    def __repr__(self):
        return '{}[data={}, shape={}]'.format(self.typename(),
            self._data, self.shape())

    def __eq__(self, other):
        return self.sameas(other)

    def _sameas(self, other):
        return array_equiv(self._data, other._data)

    def _isZero(self):
        return npla.norm(self._data, ord=np.inf)==0.0



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


class VectorInterface(ABC):

    @abstractmethod
    def __getitem__(self, i):
        pass




def Vector(x):
    '''Create a Vector expression.'''
    # if the input is a numpy array, put it into a ConstantVectorExpr
    if isinstance(x, np.ndarray):
        order = len(x.shape())
        assert(order==1)
        return ConstantVector(x)

    # if the input is a 1D list or tuple:
    #   (*) Form a ConstantVectorExpr if all the elements are constants
    #   (*) Form a VectorExpr otherwise
    if isinstance(x, (list, tuple)):
        allConsts = True
        elems = []
        for x_i in x:
            if len(x_i)!=1:
                raise ValueError('Vector() input not 1D: [{}]'.format(x))
            if isinstance(x_i, Number):
                elems.append(x_i)
            elif isinstance(x_i, Expr):
                if x_i.shape()() != ScalarShape():
                    raise ValueError('Vector() element not scalar: [{}]'.format(x))
                if not x_i.isConstant():
                    allConsts = False
                    elems.append(x_i)
                else:
                    elems.append(x_i.data())
            else:
                raise ValueError('Vector() input neither number nor expr: [{}]'.format(x_i))

        if allConsts:
            return ConstantVectorExpr(np.array(elems))
        else:
            exprElems = []
            for e in elems:
                exprElems.append(Expr._convertToExpr(e))
            return VectorExpr(e)

class ConstantVectorExpr(ConstantExprBase, VectorInterface):
    def __init__(self, data):
        shape = VectorShape(len(data))
        super(ConstantExprBase).__init__(data, shape)

    def typename(self):
        return "ConstantVector"

    def timesConstant(self, other):
        assert(isinstance(other, ConstantExprBase))

        resultShape = ExprShape.productShape(self.shape(), other.shape())

        if isinstance(resultShape, ScalarShape):
            return ConstantScalarExpr(self.data() * other.data())
        if isinstance(resultShape, VectorShape):
            return ConstantVectorExpr(np.dot(self.data(), other.data()))

    def __getitem__(self, i):
        return self.data()[i]

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
