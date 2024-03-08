from collections.abc import Iterable
from SymPDE.ExprShape import ExprShape
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import UnaryMinus, SumExpr, ProductExpr, PowerExpr, QuotientExpr
from SymPDE.DictFuncs import interdict, listInterdict

x = Coordinate(0)
y = Coordinate(1)

g = 2*x 
context = 'no context'
Petitions = ['x', ('x','y')]

g_eval = g._makeEval(context)

[Qconst, Qvar] = g_eval.buildQForOrder(1)
print("Qconst = {}, Qvar = {}".format(Qconst, Qvar))

[Aconst, Avar] = g_eval.buildAForOrder(1)
print("Aconst = {}, Avar = {}".format(Aconst, Avar))

# [Rconst, Rvar] = g_eval.buildR(Petitions)
# print("Rconst = {}, Rvar = {}".format(Rconst, Rvar))