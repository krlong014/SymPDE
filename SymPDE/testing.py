from collections.abc import Iterable
from SymPDE.ExprShape import ExprShape
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import SumExpr, ProductExpr, PowerExpr, QuotientExpr

a0 = Coordinate(0)
a1 = Coordinate(1)
g = a0 * a1

# Fsets = g.buildF(3)
# print(Fsets)

# [Qconst, Qvar] = g.buildQ()

[Aconst, Avar] = g.buildA()
print("Aconst = {}, Avar = {}".format(Aconst,Avar))