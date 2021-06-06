from . Expr import Expr
from . IndexableExpr import (
        IndexableExprIterator,
        IndexableExprInterface,
        IndexableExprElementInterface
    )
from . ExprShape import (
        ExprShape, ScalarShape, TensorShape,
        VectorShape, AggShape
    )
from . ConstantExpr import ConstantVectorExpr
from . AggExpr import AggExpr
from numpy import ndarray, array
from numbers import Number
import logging


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
        return ConstantVectorExpr(x)

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


class AggedVectorExpr(Expr, IndexableExprInterface):

    def __init__(self, elems):
        self._elems = elems
        super().__init__(VectorShape(len(elems)))

    def __getitem__(self, i):
        return self._elems[i]

    def __iter__(self):
        return AggedVectorIterator(self)

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





class AggedVectorIterator(IndexableExprIterator):
    '''Iterator for expressions stored in containers'''
    def __init__(self, parent):
        '''Constructor'''
        super().__init__(AggedVectorExpr, parent)



class AggedVectorElement(IndexableExprElementInterface):
    '''Element of an aggregated expression.'''
    def __init__(self, parent, index):
        '''Constructor'''
        super().__init__(AggedVectorExpr, parent, index)


if __name__=='__main__':
    x = Vector(1, 2)
    print('x=', x)
