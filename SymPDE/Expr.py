# =============================================================================
#
# =============================================================================
from abc import ABC, abstractmethod
from . ExprShape import (ExprShape, ScalarShape, TensorShape,
    VectorShape, AggShape)
from numbers import Number
from numpy import dot, array_equiv, inf, ndarray
import copy
import logging

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

    # Detect aggregate expressions.
    def isAggregate(self):
        '''Return true if the expression is an aggregate of expressions.'''
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
        from . ArithmeticExpr import UnaryMinus
        from . ConstantExpr import ConstantExprBase

        if isinstance(self, ConstantExprBase):
            rtn = copy.deepcopy(self)
            rtn._data = -rtn._data
            return rtn
        if self.isAggregate():
            raise ValueError('Cannot negate an Aggregate')
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

        from . ArithmeticExpr import SumExpr

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)

        logging.debug('_addOrSubtract: L=[{}], R=[{}], sign={}'.format(left, right, sign))

        # Check that neither operand is an aggregate
        if left.isAggregate() or right.isAggregate():
            raise ValueError('Cannot add/subtract aggregate expr:\nL={}\nR={}'
                .format(left, right))

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
        if sign>0:
            if left.lessThan(right):
                return SumExpr(left, right, sign)
            if right.lessThan(left):
                return SumExpr(right, left, sign)
            if (not right.lessThan(left)) and (not left.lessThan(right)):
                return 2*left
        return SumExpr(left, right, sign)


    def _multiply(leftIn, rightIn):
        '''Internal multiplication.'''

        from . ArithmeticExpr import ProductExpr, DotProductExpr
        from . ConstantExpr import ConstantVectorExpr

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)

        logging.debug('_multiply: L=[{}], R=[{}]'.format(left, right))

        # Check that neither operand is an aggregate
        if left.isAggregate() or right.isAggregate():
            raise ValueError('Cannot multiply aggregate expr:\nL={}\nR={}'
                .format(left, right))

        # Check for compatibility
        ExprShape.assertMultiplicativeCompatibility(Expr._getShape(left),
            Expr._getShape(right))

        # Zero times anything is zero
        if Expr._isZero(left) or Expr._isZero(right):
            outShape = ExprShape.productShape(left.shape(), right.shape())
            if isinstance(outShape, ScalarShape):
                return 0
            elif isinstance(outShape, VectorShape):
                return ConstantVectorExpr([0,]*outShape.dim())

        # Multiplying by the identity does nothing
        if Expr._isIdentity(right):
            return left
        if Expr._isIdentity(left):
            return right

        # If both are constant, multiply now
        if left.isConstant() and right.isConstant():
            return left.timesConstant(right)


        # expr times scalar or scalar times expr
        if (left.isScalar() or right.isScalar()):
            if right.isConstant() or right.lessThan(left):
                left,right = right,left
            return ProductExpr(left, right)

        # vector dot vector
        if (left.isVector() and right.isVector()):
            if (right.isConstant() or right.lessThan(left)):
                left,right = right,left
            return DotProductExpr(left, right)

        return NotImplementedError('Tensor-Vector multiplication not ready')


    def _divide(leftIn, rightIn):
        '''Internal division.'''

        from . ArithmeticExpr import QuotientExpr, ProductExpr
        from . ConstantExpr import ConstantScalarExpr

        # Convert input to Expr
        left = Expr._convertToExpr(leftIn)
        right = Expr._convertToExpr(rightIn)

        # Check that neither operand is an aggregate
        if left.isAggregate() or right.isAggregate():
            raise ValueError('Cannot divide aggregate expr:\nnum={}\ndenom={}'
                .format(left, right))

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

        from . ArithmeticExpr import PowerExpr
        from . ConstantExpr import ConstantScalarExpr

        # Check that neither operand is an aggregate
        if base.isAggregate() or power.isAggregate():
            raise ValueError('Cannot  exponentiate aggregate expr:\nbase={}\nexp={}'
                .format(base, power))

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
    # Comparison for ordering exprs
    # =========================================================================

    def typeLessThan(self, other):
        return type(self).__name__ < type(other).__name__

    def lessThan(self, other):
        if self.typeLessThan(other):
            return True
        if other.typeLessThan(self):
            return False
        return self._lessThan(other)

    @abstractmethod
    def _lessThan(self, other):
        pass

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

        from . ConstantExpr import ConstantExprBase

        if isinstance(x, Number) and x==0:
            return True
        if not x.isConstant():
            return False
        if isinstance(x, ConstantExprBase):
            return ConstantExprBase._isZero(x)
        return False


    def _isIdentity(x):
        '''Determine whether an argument is a multiplicative identity. '''
        from . ConstantExpr import ConstantScalarExpr

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
        from . ArithmeticExpr import SumExpr

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

        from . ConstantExpr import ConstantScalarExpr, ConstantVectorExpr

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
