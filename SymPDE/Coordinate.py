#############################################################################
#
# Class for coordinate functions (e.g., x, y, z)
#
#############################################################################

from . Expr import Expr
from . ExprShape import ScalarShape
from scipy.special import binom



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

    def buildAForOrder(self,d):
        Avar = {}
        if d == 1:
            Aconst = {self._dir : 1}
        else:
            Aconst = {}

        return Aconst, Avar



    # def buildAForOrder(self,d):
    #     if d == 1:
    #         A = [self._dir] 
    #     else:
    #         A = []

    #     if len(A) != 0:
    #         mults = [[int(binom(d,i)) for i in range(len(A))]]
    #         A = [A + mults]

    #     return A 

    def buildAllAUpToOrder(self,d):
        Asets = []
        for i in range(d):
            Asets.append(self.buildAForOrder(i+1))

        return Asets 

    # def buildA(self,d):
    #     Avar = {}
    #     Aconst = {self._dir : 1}
    #     return Aconst, Avar


