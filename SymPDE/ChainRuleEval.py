
from SymPDE.ExprEval import ExprEvaluator

class ChainRuleEvaluator(ExprEvaluator):
	'''
	ChainRuleEvaluator is the base class for evaluators of all non-leaf 
	expression nodes. 
	'''
	def __init__(self, expr, context):
		from SymPDE.ExprWithChildren import ExprWithChildren
		assert(isinstance(expr,ExprWithChildren))
		super().__init__(expr, context)
		self.childEvaluators = expr.getEvalsForChildren(context)
		
	def childEval(self, i):
		return self.childEvaluators[i]
		
class UnaryExprEvaluator(ChainRuleEvaluator):
	'''Base class for evaluators of unary expressions'''
	def __init__(self, expr, context):

		from SymPDE.ExprWithChildren import UnaryExpr
		assert(isinstance(expr,UnaryExpr))
		super().__init__(expr, context)
		
	def argEval(self):
		return self.childEval(0)
		
class BinaryExprEvaluator(ChainRuleEvaluator):
	def __init__(self, expr, context):

		from SymPDE.ExprWithChildren import BinaryExpr
		assert(isinstance(expr, BinaryExpr))
		super().__init__(expr, context)
		
	def leftEval(self):
		return self.childEval(0)
		
	def rightEval(self):
		return self.childEval(1)

class UnaryMinusEvaluator(UnaryExprEvaluator):
	def __init__(self, expr, context):

		from SymPDE.ArithmeticExpr import UnaryMinus
		assert(isinstance(expr, UnaryMinus)) 
		super().__init__(expr, context)
		
	def __str__(self):
		ae = self.argEval()
		return 'UnaryMinusEvaluator(arg={})'.format(ae.__str__())

class SumEvaluator(BinaryExprEvaluator):
	def __init__(self, expr, context):

		from SymPDE.ArithmeticExpr import SumExpr
		assert(isinstance(expr, SumExpr)) 
		super().__init__(expr, context)
		
	def __str__(self):
		L = self.leftEval()
		R = self.rightEval()
		return 'SumEvaluator(left={}, right={})'.format(L.__str__(),
                                              R.__str__())
	

