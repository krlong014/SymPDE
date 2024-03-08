import itertools as it
from scipy.special import binom
from SymPDE.DictFuncs import interdict, listInterdict
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

	#intermediate step in part
	def flatten(lst):
		if type(lst) != list:
			return lst

		flatList = []

		for item in lst:
			if type(item) == list:
				for x in flatten(item):
					flatList.append(x)
			else:
				flatList.append(item)

		return flatList

	#intermediate step in intPart
	def part(n):
		if n == 1:
			return [0]

		parts = []
		for i in range(n):
			for j in range(len(part(i))):
				parts.append(flatten([n-i] + [part(i)[j]]))

		return parts 

	#builds a list of integer partitions of an integer n
	def intPart(n):
		return [[n]] + part(n)

	#builds a dictionary with all derivatives of order d and their multiplicities
	def buildFForOrder(self,d):
		n = len(self.childEvaluators)
		if d == 1:
			F = [i for i in range(n)]
		else:
			dummyIter = [(i) for i in range(n)]
			F = list(it.combinations_with_replacement(dummyIter,d))

		mults = [int(binom(d,i)) for i in range(len(F))]
		FWithMults = {}
		for i in range(len(mults)):
			FWithMults[F[i]] = mults[i]

		return FWithMults

	#builds a dictionary with all derivatives up to order d and their multiplicities
	def buildAllFUpToOrder(self,d):
		n = len(self.childEvaluators)
		Fsets = {}
		for i in range(d):
			Fsets = Fsets | self.buildFForOrder(i+1)

		return Fsets 

	#builds a dictionary of all nonzero derivatives with respect to arguments up to order d
	def buildAllQUpToOrder(self,d):
		Qconst = {}; Qvar = {}
		for i in range(d):
			[this_Qconst, this_Qvar] = self.buildQForOrder(i+1)
			Qconst = Qconst | this_Qconst
			Qvar = Qvar | this_Qvar

		return Qconst,Qvar

	def buildAForOrder(self,d):
		num_children = len(self.childEvaluators)
		Aconst = {}; Avar = {}
		if d == 0:
			Avar = {'Identity': 1}
		if d == 1:
			[Q1const, Q1var] = self.buildQForOrder(1)
			for deriv in Q1const:
				[this_Aconst, this_Avar] = self.childEval(deriv).buildAForOrder(1)
				Aconst = Aconst | this_Aconst
				Avar = Avar | this_Avar

			for deriv in Q1var:
				[this_Aconst, this_Avar] = self.childEval(deriv).buildAForOrder(1)
				Avar = Avar | this_Avar | this_Aconst
		if d == 2:
			dummyAConsts = []
			[Q1const, Q1var] = self.buildQForOrder(1)
			[Q2const, Q2var] = self.buildQForOrder(2)

			for deriv in Q1const:
				[this_Aconst, this_Avar] = self.childEval(deriv).buildAForOrder(2)
				dummyAConsts.append(this_Aconst)
				Avar = Avar | this_Avar

			for deriv in Q1var:
				[this_Aconst, this_Avar] = self.childEval(deriv).buildAForOrder(2)
				Avar = Avar | this_Aconst | this_Avar

			for deriv in Q2const:
				[this_Aconst_0, this_Avar_0] = self.childEval(deriv[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.childEval(deriv[1]).buildAForOrder(1)
				
				this_Aconst = {}; this_Avar = {}

				this_Aconst_keys = list(zip(this_Aconst_0,this_Aconst_1))
				this_Avar_keys = list(zip(this_Avar_0,this_Avar_1))

				if len(this_Aconst_keys) == 0:
					pass
				else:
					for k in range(len(this_Aconst_keys)):
						this_Aconst[this_Aconst_keys[k]] = Q2const[deriv]
				
				if len(this_Avar_keys) == 0:
					pass
				else:
					for k in range(len(this_Avar_keys)):
						this_Avar[this_Avar_keys[k]] = Q2const[deriv]
				
				dummyAConsts.append(this_Aconst)
				Avar = Avar | this_Avar



			for deriv in Q2var:
				[this_Aconst_0, this_Avar_0] = self.childEval(deriv[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.childEval(deriv[1]).buildAForOrder(1)
				
				this_Aconst = {}; this_Avar = {}

				this_Aconst_keys = list(zip(this_Aconst_0, this_Aconst_1))
				this_Avar_keys = list(zip(this_Avar_0, this_Avar_1))

				if len(this_Aconst_keys) == 0:
					pass
				else:
					for k in range(len(this_Aconst_keys)):
						this_Avar[this_Aconst_keys[k]] = Q2var[deriv]
				
				if len(this_Avar_keys) == 0:
					pass
				else:
					for k in range(len(this_Avar_keys)):
						this_Avar[this_Avar_keys[k]] = Q2var[deriv]

				Avar = Avar | this_Avar

			if len(dummyAConsts) == 0:
				Aconst = {}
			else:
				Aconst = listInterdict(dummyAConsts)

		if d == 3:
			dummyAConsts = []
			[Q1const, Q1var] = self.buildQForOrder(1)
			[Q2const, Q2var] = self.buildQForOrder(2)
			[Q3const, Q3var] = self.buildQForOrder(3)

			for deriv in Q1const:
				[this_Aconst, this_Avar] = self.childEval(deriv).buildAForOrder(3)
				dummyAConsts.append(this_Aconst)
				Avar = Avar | this_Avar

			for deriv in Q1var:
				[this_Aconst, this_Avar] = self.childEval(deriv).buildAForOrder(3)
				Avar = Avar | this_Avar | this_Aconst

			for deriv in Q2const:
				[this_Aconst_0, this_Avar_0] = self.childEval(deriv[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.childEval(deriv[1]).buildAForOrder(2)
				
				this_Aconst = {}; this_Avar = {}

				this_Aconst_keys = list(zip(this_Aconst_0,this_Aconst_1))
				this_Avar_keys = list(zip(this_Avar_0,this_Avar_1))

				if len(this_Aconst_keys) == 0:
					pass
				else:
					for k in range(len(this_Aconst_keys)):
						this_Aconst[this_Aconst_keys[k]] = Q2const[deriv]
				
				if len(this_Avar_keys) == 0:
					pass
				else:
					for k in range(len(this_Avar_keys)):
						this_Avar[this_Avar_keys[k]] = Q2const[deriv]
				
				dummyAConsts.append(this_Aconst)
				Avar = Avar | this_Avar

			for deriv in Q2var:
				[this_Aconst_0, this_Avar_0] = self.childEval(deriv[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.childEval(deriv[1]).buildAForOrder(2)
				
				this_Aconst = {}; this_Avar = {}

				this_Aconst_keys = list(zip(this_Aconst_0, this_Aconst_1))
				this_Avar_keys = list(zip(this_Avar_0, this_Avar_1))

				if len(this_Aconst_keys) == 0:
					pass
				else:
					for k in range(len(this_Aconst_keys)):
						this_Avar[this_Aconst_keys[k]] = Q2var[deriv]
				
				if len(this_Avar_keys) == 0:
					pass
				else:
					for k in range(len(this_Avar_keys)):
						this_Avar[this_Avar_keys[k]] = Q2var[deriv]

				Avar = Avar | this_Avar

			for deriv in Q3const:
				[this_Aconst_0, this_Avar_0] = self.childEval(deriv[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.childEval(deriv[1]).buildAForOrder(1)
				[this_Aconst_2, this_Avar_2] = self.childEval(deriv[2]).buildAForOrder(1)

				this_Aconst = {}; this_Avar = {}

				this_Aconst_keys = list(zip(this_Aconst_0, this_Aconst_1, this_Aconst_2))
				this_Avar_keys = list(zip(this_Avar_0, this_Avar_1, this_Avar_2))

				if len(this_Aconst_keys) == 0:
					pass
				else:
					for k in range(len(this_Aconst_keys)):
						this_Aconst[this_Aconst_keys[k]] = Q3const[deriv]

				if len(this_Avar_keys) == 0:
					pass
				else:
					for k in range(len(this_Avar_keys)):
						this_Avar[this_Avar_keys[k]] = Q3var[deriv]

				dummyAConsts.append(this_Aconst)
				Avar = Avar | this_Avar 

			for deriv in Q3var:
				[this_Aconst_0, this_Avar_0] = self.childEval(deriv[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.childEval(deriv[1]).buildAForOrder(1)
				[this_Aconst_2, this_Avar_2] = self.childEval(deriv[2]).buildAForOrder(1)

				this_Avar = {}

				this_Aconst_keys = list(zip(this_Aconst_0, this_Aconst_1, this_Aconst_2))
				this_Avar_keys = list(zip(this_Avar_0, this_Avar_1, this_Avar_2))

				if len(this_Aconst_keys) == 0:
					pass
				else:
					for k in range(len(this_Aconst_keys)):
						this_Avar[this_Aconst_keys[k]] = Q3var[deriv]

				if len(this_Avar_keys) == 0:
					pass
				else:
					for k in range(len(this_Avar_keys)):
						this_Avar[this_Avar_keys[k]] = Q3var[deriv]

				Avar = Avar | this_Avar 

			if len(dummyAConsts) == 0:
				Aconst = {}
			else:
				Aconst = listInterdict(dummyAConsts)

		return Aconst, Avar

	def buildAllAUpToOrder(self,d):
		Aconst = {}; Avar = {}
		for i in range(d+1):
			[this_Aconst, this_Avar] = self.buildAForOrder(i)
			print("order = ", i)
			print("this_Aconst = {}, this_Avar = {}".format(this_Aconst,this_Avar))
			Aconst = Aconst | this_Aconst
			Avar = Avar | this_Avar

		return Aconst, Avar 


	def buildR(self,Petitions):
		dummyLens = []
		for item in Petitions:
			dummyLens.append(len(item))
			max_order = max(dummyLens)

		Rconst = {}; Rvar = {}

		for i in range(max_order+1):
			[Aconst, Avar] = self.buildAForOrder(i)
			print("Aconst = {}, Avar = {}".format(Aconst, Avar))
			for item in Petitions:
				if item in Aconst.keys():
					Rconst[item] = Aconst[item]
				if item in Avar.keys():
					Rvar[item] = Avar[item]

		return Rconst, Rvar

		
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

	#builds a dictionary of nonzero derivatives with respect to arguments of order d
	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)

		Qconst = {}; Qvar = {}
		if d == 1:
			Qconst = {0:1}

		return Qconst, Qvar

	def buildAForOrder(self,d):
		[Aconst, Avar] = self.argEval().buildAForOrder(d)
		
		return Aconst, Avar 

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

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)
		F_keys = F.keys()

		Qconst = {}; Qvar = {}
		const_keys_to_add = []
		if d == 1:
			for key in F_keys:
				const_keys_to_add.append(key)

		for key in const_keys_to_add:
			Qconst[key] = F[key] 

		return Qconst, Qvar

class ProductEvaluator(BinaryExprEvaluator):
	def __init__(self, expr, context):

		from SymPDE.ArithmeticExpr import ProductExpr
		assert(isinstance(expr, ProductExpr)) 
		super().__init__(expr, context)
		
	def __str__(self):
		L = self.leftEval()
		R = self.rightEval()
		return 'ProductEvaluator(left={}, right={})'.format(L.__str__(),
                                              R.__str__())

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)
		F_keys = F.keys()

		Qconst = {}; Qvar = {}
		if d == 1:
			Qvar = F 
		elif d == 2:
			const_keys_to_add = []
			for key in F_keys:
				if key[0] != key[1]:
					const_keys_to_add.append(key)

			for key in const_keys_to_add:
				Qconst[key] = F[key] 

		return Qconst, Qvar 

class QuotientEvaluator(BinaryExprEvaluator):
	def __init__(self,expr,context):
		from SymPDE.ArithmeticExpr import QuotientExpr
		assert(isinstance(expr,QuotientExpr))
		super().__init__(expr,context)

	def __str__(self):
		L = self.leftEval()
		R = self.rightEval()
		return 'QuotientEvaluator(numerator={}, denominator={})'.format(L.__str__(),
                                              R.__str__())

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)
		F_keys = F.keys()

		Qconst = {}; Qvar = F 
		if d >= 2:
			keys_to_remove = []
			for deriv in Qvar:
				num_zeros = deriv.count(0)
				if num_zeros >= 2:
					keys_to_remove.append(deriv)

			for key in keys_to_remove:
				del Qvar[key]

		return Qconst, Qvar

class PowerEvaluator(BinaryExprEvaluator):
	def __init__(self,expr,context):
		from SymPDE.ArithmeticExpr import PowerExpr
		assert(isinstance(expr, PowerExpr))
		super().__init__(expr,context)

	def __str__(self):
		L = self.leftEval()
		R = self.rightEval()
		return 'PowerEvaluator(base={}, exponent={})'.format(L.__str__(),
                                              R.__str__())

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)

		Qconst = {}; Qvar = F 
		return Qconst, Qvar 



