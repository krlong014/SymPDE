
from SymPDE.ExprEval import ExprEvaluator

class ChainRuleEvaluator(ExprEvaluator):
	def __init__(self, expr, context):
		from SymPDE.ExprWithChildren import ExprWithChildren
		assert(isinstance(expr,ExprWithChildren))
		self.childEvaluators = expr.getEvalsForChildren(context)
		
	def childEval(self, i):
		return self.childEvaluators[i]
		
class UnaryExprEvaluator(ChainRuleEvaluator):
	def __init__(self, expr, context):

		from SymPDE.ExprWithChildren import UnaryExpr
		assert(isinstance(expr,UnaryExpr))
		super().__init__(expr, context)
		
	def argEval(self):
		return self.childEval(0)

class UnaryMinusEvaluator(UnaryExprEvaluator):
	def __init__(self, expr, context):

		from SymPDE.ArithmeticExpr import UnaryMinus
		assert(isinstance(expr, UnaryMinus)) 
		super().__init__(expr, context)
		
	def __str__(self):
		ae = self.argEval()
		return 'UnaryMinusEvaluator(arg={})'.format(ae.__str__())