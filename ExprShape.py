

class ExprShape:
    def __init__(self, dim):
        self.dim = dim


    def sameas(self, other):
        return (type(self)==type(other) and self.dim==other.dim)


    def additionCompatible(left, right):
        '''Determine whether two operands are additively compatible'''

        if isinstance(left, ListShape) or isinstance(right, ListShape):
            return False

        # Make sure both inputs are subtypes of ExprShape
        if not isinstance(left, ExprShape):
            raise TypeError(
                'invalid left operand to mult compatibility test [{}]'.format(left)
                )

        if not isinstance(left, ExprShape):
            raise TypeError(
                'invalid left operand to mult compatibility test [{}]'.format(left)
                )

        return left==right

    def multiplicationCompatible(left, right):
        '''Determine whether two operands are multiplicatively compatible'''

        if isinstance(left, ListShape) or isinstance(right, ListShape):
            return False

        # Make sure both inputs are subtypes of ExprShape
        if not isinstance(left, ExprShape):
            raise TypeError(
                'invalid left operand to mult compatibility test [{}]'.format(left)
                )

        if not isinstance(left, ExprShape):
            raise TypeError(
                'invalid left operand to mult compatibility test [{}]'.format(left)
                )

        # Anything times a scalar is defined
        if isinstance(left,ScalarShape) or isinstance(right, ScalarShape):
            return True

        # If dimensions are not equal, there is no compatibility
        if left.dim != right.dim:
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
        if not ExprShape.additionCompatible(left, right):
            raise ValueError('''operands left={} and right={} not compatible for
                                addition'''.format(left, right))

    def assertMultiplicativeCompatibility(left, right):
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
            return VectorShape(right.dim)


        # Tensor times vector produces a column vector
        if (isinstance(left, VectorShape) and isinstance(right, TensorShape)):
            return VectorShape(right.dim)

        # Tensor times tensor produces a tensor
        if (isinstance(left, TensorShape) and isinstance(right, TensorShape)):
            return TensorShape(right.dim)

        # Row vector times a column vector produces a scalar
        if (isinstance(left, VectorShape) and isinstance(right, VectorShape)):
            return ScalarShape()

        raise RuntimeError('Impossible case in productShape()')

    def __ne__(self, other):
        return not (self==other)


class ListShape(ExprShape):
    def __init__(self):
        super().__init__(-1)

    def __eq__(self, other):
        return isinstance(other, ListShape)

    def __str__(self):
        return "List"




class ScalarShape(ExprShape):
    def __init__(self):
        super().__init__(1)

    def __eq__(self, other):
        return isinstance(other, ScalarShape)

    def __str__(self):
        return "Scalar"


class VectorShape(ExprShape):
    def __init__(self, dim):
        super().__init__(dim)

    def __eq__(self, other):
        return (isinstance(other, VectorShape)
                and other.dim==self.dim)

    def __str__(self):
        return "Vector(dim=%d)" % self.dim


class TensorShape(ExprShape):
    def __init__(self, dim):
        super().__init__(dim)

    def __eq__(self, other):
        return (isinstance(other, TensorShape) and other.dim==self.dim)

    def __str__(self):
        return "Tensor(dim=%d)" % self.dim
