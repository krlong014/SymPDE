###############################################################################
# Abstract interface for expressions with that can be indexed, and for
# elements thereof.
###############################################################################
from abc import ABC, abstractmethod


class IndexableExprInterface(ABC):
    '''Abstract interface for expressions that can be indexed.'''
    @abstractmethod
    def __getitem__(self, i):
        pass

    @abstractmethod
    def __iter__(self):
        pass



class IndexableExprIterator:
    '''Iterator for elements of expressions'''
    def __init__(self, objectType, parent):
        '''Constructor'''
        self._index = 0
        assert(isinstance(parent, objectType))
        self._parent = parent

    def __next__(self):
        '''Advance the iterator'''
        if self._index>=0 and self._index < len(self._parent):
            result = self._parent._elems[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration


class IndexableExprElementInterface:
    '''Abstract interface for elements of indexable expressions.'''
    def __init__(self, objectType, parent, index):
        assert(isinstance(parent, objectType))
        self._parent = parent
        self._index = index

    def parent(self):
        return self._parent

    def index(self):
        return self._index

    def __repr__(self):
        return 'Element[{}] of expression [{}]'.format(
            self.index(), self.parent())

    def __str__(self):
        return '{}[{}]'.format(self.parent(), self.index())
