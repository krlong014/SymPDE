from Expr import Expr

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
            assert(not c.isAggregate())

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

    def _lessThan(self, other):
        if len(self._children) < len(other._children):
            return True
        if len(self._children) > len(other._children):
            return False

        for mine, yours in zip(self._children, other._children):
            if mine.lessThan(yours):
                return True
            if yours.lessThan(mine):
                return False
        return False

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

    @abstractmethod
    def opString(self):
        pass

    def __str__(self):
        return '{}{}{}'.format(Expr._maybeParenthesize(self.left()),
            self.opString(), Expr._maybeParenthesize(self.right()))

    def isSpatialConstant(self):
        return self.left().isSpatialConstant() and self.right().isSpatialConstant()

class SumExpr(BinaryExpr):
    def __init__(self, L, R, sign):
        super().__init__(L, R, Expr._getShape(L))
        self.sign = sign

    def opString(self):
        if self.sign > 0:
            return '+'
        else:
            return '-'

    def _sameas(self, other):
        return (self.left().sameas(other.left()) and self.right().sameas(other.right())
            and self.sign==other.sign)

    def _lessThan(self, other):
        if self.sign < other.sign:
            return True
        if self.sign > other.sign:
            return False
        return super()._lessThan(other)

    def __repr__(self):
        return 'SumExpr[left={}, right={}, shape={}]'.format(self.left().__repr__(),
            self.right().__repr__(), self.shape())


class ProductExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())

    def opString(self):
        return '*'

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

    def opString(self):
        return 'dot'

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

    def opString(self):
        return 'cross'

    def __str__(self):
        return 'cross({},{})'.format(self.L, self.R)

class QuotientExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, L.shape())

    def opString(self):
        return '/'



class PowerExpr(BinaryExpr):
    def __init__(self, L, R):
        super().__init__(L, R, ScalarShape())

    def opString(self):
        return 'power'

    def __str__(self):
        return 'pow({},{})'.format(self.left(), self.right)



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
        if isinstance(self.shape(), ScalarShape):
            return self._data == 0.0
        return norm(self._data, ord=inf)==0.0



class ConstantScalarExpr(ConstantExprBase):
    def __init__(self, data):
        super().__init__(data, ScalarShape())

    def typename(self):
        return "ConstantScalar"

    def _lessThan(self, other):
        return self.data() < other.data()

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
        return self._parent

    def index(self):
        return self._index

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

    def _lessThan(self, other):
        if len(self) < len(other):
            return True
        if len(self) > len(other):
            return False
        for (mine, yours) in zip(self.data(), other.data()):
            if mine<yours:
                return True
            if yours<mine:
                return False
        return False





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

    def _lessThan(self, other):
        return self._dir < other._dir

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

    def _lessThan(self, other):
        if len(self) < len(other):
            return True
        if len(self) > len(other):
            return False

        for (mine, yours) in zip(self, other):
            if mine.lessThan(yours):
                return True
            if yours.lessThan(mine):
                return False
        return False




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
