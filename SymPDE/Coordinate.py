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

    def name(self):
        return self._name
    
    def _makeEval(self, context):
        return CoordinateEvaluator(self, context)

    # def buildAForOrder(self,d):
    #     Avar = {}
    #     if d == 1:
    #         Aconst = {self._dir : 1}
    #     else:
    #         Aconst = {}

    #     return Aconst, Avar



from . ExprEval import ExprEvaluator

class CoordinateEvaluator(ExprEvaluator):
    
    def __init__(self, expr, context):
        super().__init__(expr,context)

    def __str__(self):
        return 'CoordEvaluator({})'.format(self.myExpr().name())
    

    def buildAForOrder(self,d):
        Aconst = {}; Avar = {}

        if d == 1:
            Aconst[self.myExpr().name()] = 1

        return Aconst, Avar 


