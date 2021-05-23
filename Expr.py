# =============================================================================
#
# =============================================================================
from abc import ABC, abstractmethod
from ExprShape import (ExprShape, ScalarShape, TensorShape,
    VectorShape, AggShape)
from numbers import Number
from numpy.linalg import norm
from numpy import dot, array_equiv, inf, ndarray
import copy

class Expr(ABC):
    '''Expr is the base class for symbolic expressions.'''
    def __init__(self, shape):
        '''Construct an expression of a specified shape.'''
        assert(isinstance(shape, ExprShape))
        self._shape = shape

    # Describe the structure ('shape') of the expression: scalar,
    # vector, tensor, or list.
    def shape(self):
        '''Return the shape (scalar, vector, tensor) of the expression.'''
        return self._shape

    def isScalar(self):
        '''Return true if is scalar, false otherwise'''
        return isinstance(self.shape(), ScalarShape)

    def isVector(self):
        '''Return true if is vector, false otherwise'''
        return isinstance(self.shape(), ScalarShape)

    def __len__(self):
        '''Return the dimension:
        1 for scalars, spatial dim for vectors/tensors.'''
        return self.shape().dim()

    # Compare contents with another expression. The sameas() function is used
    # to indicate identical contents rather than identical objects. The
    # top-level sameas() function compares type and shape. If type and shape
    # are the same, then the type's internal _sameas() function is called
    # to do a detailed comparison of contents.
    def sameas(self, other):
        '''Return true if self and other have same contents.'''
        return (type(self)==type(other)
            and self.shape().sameas(other.shape())
            and self._sameas(other))

    @abstractmethod
    def _sameas(self, other):
        '''Compare to another expression that can be assumed to be of the same type.'''
        pass

    # Detect constant expression.
    def isConstant(self):
        '''Indicate whether an expression is constant, i.e., independent
        of space and unable to be changed by the user. A Parameter is not
        a constant.'''
        return False

    # Detect spatially constant expressions. This includes not only numerical
    # constants but parameters that might change (e.g., design parameters,
    # simulation time) but don't depend on space.
    def isSpatialConstant(self):
        '''Indicate whether an expression does not depend on space.'''
        return False

    # =========================================================================
    # Overloaded operators
    # Note: is __op__() is a binary operation, then __rop__() is used when the
    # left-hand operand is a basic type and not an object.
    #
    # With the exception of the negation operator, these functions are
    # thin interfaces to internal functions.
    # =========================================================================

    def __neg__(self):
        '''Negate self.'''
        if isinstance(self, ConstantExprBase):
            rtn = copy.deepcopy(self)
            rtn._data = -rtn._data
            return rtn
        if isinstance(self, AggExpr):
            raise ValueError('cannot negate a AggExpr')
        return UnaryMinus(self)

    def __add__(self, other):
        '''Add self + other. Expressions must have same shape.'''
        return Expr._addOrSubtract(self, other, 1)

    def __sub__(self, other):
        '''Subtract self - other. Expressions must have same shape.'''
        return Expr._addOrSubtract(self, other, -1)


    def __mul__(self, other):
        '''Multiply self * other. Expressions must have compatible shapes for
        scalar-scalar multiplication,
        vector-scalar and scalar-vector multiplication, the
        vector dot product, vector-tensor and tensor-vector products.'''
        return Expr._multiply(self, other)

    def __truediv__(self, other):
        '''Divide self/other. Denominator must be a scalar.'''
        return Expr._divide(self, other)

    def __pow__(self, other):
        '''Raise self to the power other. Both must be scalars.'''
        return Expr._power(self, other)

    def __radd__(self, other):
        '''Add other + self, when 'other' is a number.'''
        return Expr._addOrSubtract(other, self, 1)

    def __rsub__(self, other):
        '''Subtract other - self, when 'other' is a number.'''
        return Expr._addOrSubtract(other, self, -1)


    def __rmul__(self, other):
        '''Multiply other * self, when 'other' is a number.'''
        return Expr._multiply(other, self)

    def __rtruediv__(self, other):
        '''Divide other/self, when 'other' is a number.'''
        return Expr._divide(other, self)

    def __rpow__(self, other):
        '''Raise 'other' to the power 'self', when 'other' is a number.'''
        return Expr._power(other, self)


    # =========================================================================
    # Internal arithmetic operations. The overloaded operators delegate to
    # these. In these functions: operands are checked for compatibiity,
    # possible simplifications are detected and used, the shape of the output
    # is determined, and then the resulting expression is returned.
    # =========================================================================

    def _addOrSubtract(leftIn, rightIn, sign):
        '''Internal addition/subtraction.'''
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
        '''Internal multiplication.'''
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
        '''Internal division.'''
        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)


        # Division by a list is an error
        if isinstance(right, AggExpr):
            raise ValueError(
                'Division by list is undefined: num={}, den={}'.format(left,right))

        # Division of a list is an error
        if isinstance(left, AggExpr):
            raise ValueError(
                'Dividing a list is undefined: num={}, den={}'.format(left,right))

        # Division by a vector or tensor is an error
        if not isinstance(Expr._getShape(right),ScalarShape):
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
        '''Internal exponentiation.'''
        # Convert input to Expr
        base = Expr._convertToExpr(baseIn)
        power = Expr._convertToExpr(powerIn)

        # Exponentiation with non-scalars is undefined here
        if (not isinstance(Expr._getShape(base), ScalarShape)):
            raise ValueError('Non-scalar base [{}] in Expr._power'.format(base))

        if (not isinstance(Expr._getShape(power), ScalarShape)):
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


    # =========================================================================
    # Miscellaneous internal maintenance functions.
    # =========================================================================

    def _getShape(x):
        '''Return the shape of the argument x. '''
        if isinstance(x, Expr):
            return x.shape()
        elif isinstance(x, Number):
            return ScalarShape()
        raise ValueError('Expr._getShape() got bad arg [{}]'.format(x))


    def _isConstant(x):
        '''Determine whether an argument is a constant. '''
        return ((isinstance(x, Expr) and x.isConstant())
                or isinstance(x, Number))

    def _isZero(x):
        '''Determine whether an argument is identically zero. '''
        if isinstance(x, Number) and x==0:
            return True
        elif not x.isConstant():
            return False
        return x._isZero()
        raise ValueError('Expr._isZero() got bad arg [{}]'.format(x))

    def _isIdentity(x):
        '''Determine whether an argument is a multiplicative identity. '''
        if not x.isConstant():
            return False
        if isinstance(x, ConstantScalarExpr):
            return x.data()==1.0
        # if isinstance(x, ConstantTensorExpr):
        #     d = x.data()
        #     for i in range(x.shape().dim()):
        #         for j in range(x.shape().dim()):
        #             if ((i!=j and d[i,j] != 0.0) or (i==j and d[i,j] != 1.0)):
        #                 return False
        #     return True
        return False


    def _maybeParenthesize(x):
        '''Put a sum inside parentheses when printing. '''
        if isinstance(x, SumExpr):
            return '({})'.format(x)
        return '{}'.format(x)


    def _dirName(dir):
        '''Return the default names of the specified coordinate. '''
        cart = ('x', 'y', 'z')
        return cart[dir]


    def _convertToExpr(x):
        '''Convert input to an expression, if possible. If the input is an
        expression, return it unmodified.'''
        if isinstance(x, Expr):
            return x

        if isinstance(x, Number):
            return ConstantScalarExpr(x)

        if isinstance(x, ndarray):
            return ConstantVectorExpr(x)

        raise ValueError('input [{}] cannot be converted to Expr'.format(x))

    def _convertibleToExpr(x):
        '''Indicate whether the argument is of a type that can be
        converted to an expression. '''
        return isinstance(x, (Expr, Number, ndarray))





#############################################################################
#
# Fundamental Expr subtypes for arithmetic operations
#
#############################################################################

class ExprWithChildren(Expr):
    '''ExprWithChildren is a base class for operations that act on other
    operations.'''
    def __init__(self, children, shape):
        '''Constructor for ExprWithChildren.'''
        if not (isinstance(children, (list, tuple)) and len(children)>0):
            raise ValueError('ExprWithChildren bad input {}'.format(children))
        for c in children:
            assert(not isinstance(c, AggExpr))

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
    assert(a.shape().dim() == b.shape().dim())

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
    assert(a.shape().dim() == b.shape().dim())

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
        assert(i>=0 and i<self.shape().dim())
        return self.data()[i]

    def __len__(self):
        return self.shape().dim()





#############################################################################
#
# Class for coordinate functions (e.g., x, y, z)
#
#############################################################################


class Coordinate(Expr):

    def __init__(self, dir, name=None):
        super().__init__(ScalarShape())
        self._dir = dir
        if name==None:
            self._name = Expr._dirName(dir)
        else:
            self._name = name

    def __str__(self):
        return self._name

    def __repr__(self):
        return 'Coordinate[dir={}, name={}, shape={}]'.format(self._dir,
            self._name, self.shape())

    def _sameas(self, other):
        return self._dir==other._dir and self._name==other._name

    def direction(self):
        return self._dir


#############################################################################
#
# Class for aggregate expressions
#
#############################################################################

class AggExpr(Expr):

    def __init__(self, *args):
        super().__init__(AggShape())

        if len(args)==1:
            input = args[0]
        else:
            input = args


        if not isinstance(input, (list, tuple, Expr, Number, ndarray)):
            raise ValueError('input [{}] not convertible to AggExpr'.format(input))

        if isinstance(input, AggExpr):
            self.data = input.data
        elif isinstance(input, (list, tuple)):
            self.data = []
            for i,e in enumerate(input):
                if isinstance(e, AggExpr):
                    raise ValueError('Agg within list detected in entry \
                    #{}=[]'.format(i, e))
                if not Expr._convertibleToExpr(e):
                    raise ValueError('Agg entry #{}=[] not convertible \
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
        return AggExprIterator(self)

    def __str__(self):
        rtn = 'Agg('
        for i,e in enumerate(self.data):
            if i>0:
                rtn += ', '
            rtn += e.__str__()
        rtn += ')'
        return rtn

    def append(self, entry):
        self.data.append(entry)



class AggExprIterator:
    '''Iterator for expressions stored in containers'''
    def __init__(self, parent):
        '''Constructor'''
        self._index = 0
        assert(isinstance(parent, AggExpr))
        self._parent = parent

    def __next__(self):
        '''Advance the iterator'''
        if self._index>=0 and self._index < len(self._parent):
            result = self._parent.data[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
