
#############################################################################
#
# Class for aggregate expressions
#
#############################################################################

import Expr
from IndexableExpr import (
        IndexableExprIterator,
        IndexableExprInterface,
        IndexableExprElementInterface
    )


class AggExpr(IndexableExprInterface):

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

    def isAggregate(self):
        return True



class AggExprIterator(IndexableExprIterator):
    '''Iterator for expressions stored in containers'''
    def __init__(self, parent):
        '''Constructor'''
        super().__init__(AggExpr, parent)



class AggExprElement(IndexableExprElementInterface):
    '''Element of an aggregated expression.'''
    def __init__(self, parent, index):
        '''Constructor'''
        super().__init__(AggExpr, parent, index)
