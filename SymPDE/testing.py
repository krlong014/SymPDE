from collections.abc import Iterable
from SymPDE.ExprShape import ExprShape
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import SumExpr, ProductExpr, PowerExpr, QuotientExpr
import SymPDE.FunctionWithBasis as fwb
from SymPDE.BasisBase import BasisBase, ScalarBasisBase
from MakeEval import makeEval 
# from SymPDE.ExprEval import ExprWithChildrenEval

x = Coordinate(0)
y = Coordinate(1)

g = -x
g_eval = makeEval(g)
# Qconst, Qvar = g.buildQForOrder(1)
# print("Qconst = {}, Qvar = {}".format(Qconst,Qvar))

#we need a serparate entity that builds an 
	#evaluator based on the expression.

	#something like 
	#g_eval = makeEval(g)

	#then we can just write g_eval.buildQ