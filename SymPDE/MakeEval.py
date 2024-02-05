#this guy will have a dictionary whose keys are expression subtypes and whose values are evaluator subtypes.

#when called on an expression, it will match it to the corresponding evaluator. 

from SymPDE.Expr import Expr 
from SymPDE.ArithmeticExprEval import UnaryMinusEval
from SymPDE.ExprEval import ExprEval

def makeEval(g):
	assert(isinstance(g,Expr))

	EXPR_TO_EVAL = typetable()
	print(EXPR_TO_EVAL)

	evaluator_type = EXPR_TO_EVAL[g.myType()]
	g_eval = evaluator_type(g)
	return g_eval

def typetable():
	
	EXPR_TO_EVAL = {
		"UnaryMinus" : UnaryMinusEval
	}
	return EXPR_TO_EVAL