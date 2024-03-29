# =============================================================================
#
# =============================================================================

class ExprShape:
    '''ExprShape is the base class for specification of the structure ('shape')
    of an expression. An expression's shape will be one of
    (*) ScalarShape
    (*) VectorShape
    (*) TensorShape
    (*) Aggregate (Agg) shape

    These are implemented as subclasses of ExprShape. 

    The meaning of scalar, vector, and tensor is obvious. An aggregate is a list of expressions, 
    intended to represent collections of like variables such as concentrations of a set of species. 
    '''
    def __init__(self, dim):
        self._dim = dim

    def dim(self):
        '''If scalar, return 1. If vector or tensor, return the spatial
        dimension. If an aggregate, return -1.'''
        return self._dim


    def sameas(self, other):
        '''Indicate whether this shape is identical to another. '''
        return (type(self)==type(other) and self.dim()==other.dim())


    def additionCompatible(left, right):
        '''Determine whether two operands are additively compatible'''

        if isinstance(left, AggShape) or isinstance(right, AggShape):
            return False # HUH???

        # Make sure both inputs are subtypes of ExprShape
        if not isinstance(left, ExprShape):
            raise TypeError(
                'invalid left operand to addition compatibility test [{}]'.format(left)
                )

        if not isinstance(right, ExprShape):
            raise TypeError(
                'invalid right operand to addition compatibility test [{}]'.format(right)
                )

        return left.sameas(right)

    def multiplicationCompatible(left, right):
        '''Determine whether two operands are multiplicatively compatible'''

        if isinstance(left, AggShape) or isinstance(right, AggShape):
            return False # HUH???

        # Make sure both inputs are subtypes of ExprShape
        if not isinstance(left, ExprShape):
            raise TypeError(
                'invalid left operand to mult compatibility test [{}]'.format(left)
                )

        if not isinstance(right, ExprShape):
            raise TypeError(
                'invalid right operand to mult compatibility test [{}]'.format(right)
                )

        # Anything times a scalar is defined
        if isinstance(left,ScalarShape) or isinstance(right, ScalarShape):
            return True

        # If dimensions are not equal, there is no compatibility
        if left.dim() != right.dim():
            return False

        # Tensor times either tensor or vector is defined
        if (isinstance(left, TensorShape)
                and isinstance(right, (VectorShape, TensorShape))):
            return True

        # Vector times tensor or vector is defined
        if (isinstance(left, VectorShape)
                and isinstance(right, (VectorShape, TensorShape))):
            return True

        # Other cases not defined
        return False

    def assertAdditiveCompatibility(left, right):
        '''Throw an error if shapes are not compatible for addition.'''
        if not ExprShape.additionCompatible(left, right):
            raise ValueError('''operands left={} and right={} not compatible for
                                addition'''.format(left, right))

    def assertMultiplicativeCompatibility(left, right):
        '''Throw an error if shapes are not compatible for multiplication.'''
        if not ExprShape.multiplicationCompatible(left, right):
            raise ValueError('''operands left={} and right={} not compatible for
                                multiplication'''.format(left, right))

    def productShape(left, right):
        '''Find the Shape (scalar, vector, tensor) of a product'''

        # First make sure the operation is even defined
        ExprShape.assertMultiplicativeCompatibility(left, right)

        # Multiplication by a scalar does not change Shape
        if isinstance(left, ScalarShape):
            return right

        if isinstance(right, ScalarShape):
            return left

        # Tensor times vector produces a column vector
        if (isinstance(left, TensorShape) and isinstance(right, VectorShape)):
            return VectorShape(right.dim())


        # Tensor times vector produces a row vector
        if (isinstance(left, VectorShape) and isinstance(right, TensorShape)):
            return VectorShape(right.dim())

        # Tensor times tensor produces a tensor
        if (isinstance(left, TensorShape) and isinstance(right, TensorShape)):
            return TensorShape(right.dim())

        # Row vector times a column vector produces a scalar
        if (isinstance(left, VectorShape) and isinstance(right, VectorShape)):
            return ScalarShape()

        raise RuntimeError('Impossible case in productShape()')


#############################################################################
#
# Shape subtypes
#
#############################################################################


class AggShape(ExprShape):
    '''Class for Aggregate expression shape'''
    def __init__(self, dim):
        super().__init__(dim)

    def __str__(self):
        return "Agg"

class ScalarShape(ExprShape):
    '''Class for Scalar expression shape'''
    def __init__(self):
        super().__init__(1)

    def __str__(self):
        return "Scalar"


class VectorShape(ExprShape):
    '''Class for Vector expression shape'''
    def __init__(self, dim):
        super().__init__(dim)

    def __str__(self):
        return "Vector(dim=%d)" % self.dim()


class TensorShape(ExprShape):
    '''Class for Tensor expression shape'''
    def __init__(self, dim):
        super().__init__(dim)

    def __str__(self):
        return "Tensor(dim=%d)" % self.dim()
