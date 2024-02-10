from SymPDE.Coordinate import Coordinate, CoordinateEvaluator
from SymPDE.ArithmeticExpr import UnaryMinus
from SymPDE.ChainRuleEval import UnaryMinusEvaluator

f = Coordinate(0)
g = -f

context = 'bob'

eval = g.makeEval(context)

print('\n\nEvaluator for {} is {}\n\n'.format(f, eval))