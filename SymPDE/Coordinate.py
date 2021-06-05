#############################################################################
#
# Class for coordinate functions (e.g., x, y, z)
#
#############################################################################

import Expr
from Expr import ScalarShape


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
