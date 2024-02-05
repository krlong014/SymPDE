from SymPDE.Expr import Expr
from SymPDE.ArithmeticExpr import ExprWithChildren, UnaryMinus, UnaryExpr
from SymPDE.Coordinate import Coordinate
import itertools as it
from SymPDE.DictFuncs import interdict
from SymPDE.ExprShape import ScalarShape, VectorShape
from SymPDE.ExprEval import ExprEval
from SymPDE.MakeEval import makeEval

class ExprWithChildrenEval(ExprEval):
	def __init__(self,expr):
		assert(isinstance(expr,ExprWithChildren))
		self._children = expr.children()


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

	#builds an integer partition of n
	def intPart(n):
		return [[n]] + part(n)


	#builds a particular Q set of order d
	#appends a list of the multiplicities of each derivative to the end
	def buildFForOrder(self,d):
		n = len(self._children)
		if d == 1:
			F = [i for i in range(n)]
		else:
			dummyIter = [(i) for i in range(n)]
			F = list(it.combinations_with_replacement(dumyIter,d))

		mults = [int(binom(d,i)) for i in range(len(F))]

		FwithMults = {}
		for i in range(len(mults)):
			FwithMults[F[i]] = mults[i]

		return FwithMults

	#builds all F sets up to oder d
	def buildAllFUpToOrder(self,d):
		n = len(sef._children)
		assert(n >= 1)

		Fsets = {}
		for i in range(d):
			Fsets = Fsets | self.buildFForOrder(i+1)

		return Fsets 

	#builds all Q sets up to order d
	def buildAllQUpToOrder(self,d):
		Qconst = {}; Qvar = {}
		for i in range(d):
			[this_Qconst, this_Qvar] = self.buildQForOrder(i+1)
			Qconst = Qconst | this_Qconst
			Qvar = Qvar | this_Qvar

		return(Qconst, Qvar)

	#builds A set for order d
	def buildAForOrder(self,d):
		n = len(self._children)
		assert (n >= 1)

		if d == 1:
			[Qconst_order_1, Qvar_order_1] = self.buildQForOrder(1)

			Aconst = {}; Avar = {}
			for q in Qconst_order_1:
				[this_Aconst, this_Avar] = self.child(q).buildAForOrder(1)
				Aconst = Aconst | this_Aconst
				Avar = Avar | this_Avar

			for q in Qvar_order_1:
				[this_Aconst, this_Avar] = self.child(q).buildAForOrder(1)
				Avar = Avar | this_Aconst | this_Avar

		if d == 2:
			[Qconst_order_1, Qvar_order_1] = self.buildQForOrder(1)

			A2_term_const = {}; A2_term_var = {}
			for q in Qconst_order_1:
				[this_Aconst, this_Avar] = self.child(q).buildAForOrder(2)
				A2_term_const = A2_term_const | this_Aconst
				A2_term_var = A2_term_var | this_Avar

			for q in Qvar_order_1:
				[this_Aconst, this_Avar] = self.child(q).buildAForOrder(2)
				A2_term_var = A2_term_var | this_Aconst | this_Avar

			[Qconst_order_2, Qvar_order_2] = self.buildQForOrder(2)

			A1_term_const = {}; A1_term_var = {}
			for q in Qconst_order_2:
				[this_Aconst_0, this_Avar_0] = self.child(q[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.child(q[1]).buildAForOrder(1)
				this_Aconst = list(zip(this_Aconst_0, this_Aconst_1))
				this_Avar = list(zip(this_Avar_0, this_Avar_1))

				for item in this_Aconst:
					A1_term_const[item] = Qvar_order_2[item]
				for item in this_Avar:
					A1_term_var[item] = Qvar_order_2[item]

			for q in Qvar_order_2:
				[this_Aconst_0, this_Avar_0] = self.child(q[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.child(q[1]).buildAForOrder(1)
				this_Aconst = list(zip(this_Aconst_0, this_Aconst_1))
				this_Avar = list(zip(this_Avar_0, this_Avar_1))

				for item in this_Aconst:
					A1_term_var[item] = Qvar_order_2[item]
				for item in this_Avar:
					A1_term_var[item] = Qvar_order_2[item]

			Aconst = A1_term_const | A2_term_const
			Avar = A1_term_var | A2_term_var

		if order == 3:
			[Qconst_order_1, Qvar_order_1] = self.buildQForOrder(1)
			Q1 = Qconst_order_1 | Qvar_order_1

			A3_term_const = {}; A3_term_var = {}
			for q in Qconst_order_1:
				[this_Aconst, this_Avar] = self.child(q).buildAForOrder(3)
				A3_term_const = A3_term_const | this_Aconst
				A3_term_var = A3_term_var | this_Avar 

			[Qconst_order_2, Qvar_order_2] = self.buildQForOrder(2)
			Q2 = Qconst_order_2 | Qvar_order_2

			A2_term_const = {}; A2_term_var = {}
			for q in Qconst_order_2:
				[this_Aconst_0, this_Avar_0] = self.child(q[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.child(q[1]).buildAForOrder(2)
				this_Aconst = this_Aconst_0 | this_Aconst_1 
				this_Avar = this_Avar_0 | this_Avar_1

				for item in this_Aconst:
					A2_term_const[item] = Q2[item]
				for item in this_Avar:
					A2_term_var[item] = Q2[item]

			[Qconst_order_3, Qvar_order_3] = self.buildQForOrder(3)
			Q3 = Qconst_order_3 | Qvar_order_3

			A1_term_const = {}; A1_term_var = {}
			for q in Qconst_order_3:
				[this_Aconst_0, this_Avar_0] = self.child(q[0]).buildAForOrder(1)
				[this_Aconst_1, this_Avar_1] = self.child(q[1]).buildAForOrder(1)
				[this_Aconst_2, this_Avar_2] = self.child(q[2]).buildAForOrder(1)
				this_Aconst = list(zip(this_Aconst_0,this_Aconst_1,this_Aconst_2))
				this_Avar = list(zip(this_Avar_0,this_Avar_1,this_Avar_2))

				for item in this_Aconst:
					A1_term_const[item] = Q3[item]
				for item in this_Avar:
					A1_term_var[item] = Q3[item]

			A1_term_var = interdict(Q3,A1_term_const)

			Aconst = A3_term_const | A2_term_const | A1_term_const
			Avar = A3_term_var | A2_term_var | A1_term_var

		return Aconst, Avar

	#builds all A sets up to order d
	def buildAllAUpToOrder(self,d):
		Aconst = {}; Avar = {}
		for i in range(d):
			[this_Aconst, this_Avar] = self.buildAForOrder(i+1)
			Aconst = Aconst | this_Aconst
			Avar = Avar | this_Avar
		return Aconst, Avar 

class UnaryExprEval(ExprWithChildrenEval):
	def __init__(self,expr):
		assert(isinstance(expr,UnaryExpr))
		super().__init__(expr)

class UnaryMinusEval(UnaryExprEval):
	def __init__(self,expr):
		assert(isinstance(expr, UnaryMinus))
		super().__init__(expr)
		self.argEval = makeEval(expr.arg())

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)

		Qconst = {}; Qvar = {}
		if d == 1: Qconst = {self.arg:1}

		return Qconst, Qvar 

	def buildA(self,d):
		[Aconst, Avar] = self.arg.buildA()
		return Aconst, Avar

	def printQ(self,d):
		[Qconst, Qvar] = self.buildQ(d)

		for Q in Qconst:
			print("Qconst contains the derivative with respect to argument {}, which has multiplicity {}".format(self.arg(Q)),Qconst[Q])
		for Q in Qvar:
			print("Qvar contains the derivative with respect to argument {}, which has multiplicity {}".format(Q,Qvar[Q]))

class BinaryExprEval(ExprWithChildrenEval):
	def __init__(self, L, R, shape):
		super().__init__((L,R),shape)

class BinaryArithmeticOpEval(BinaryExprEval):
	def __init__(self,L,R,shape):
		super().__init__(L,R,shape)

class SummExprEval(BinaryArithmeticOpEval):
	def __init__(self,L,R,sign):
		super().__init__(L,R,Expr._getShape(L))
		self.L = L 
		self.R = R
		self.sign = sign 

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

	def printQ(self,d):
		[Qconst, Qvar] = self.buildQ(d)

		for Q in Qconst:
			print("Qconst contains the derivative with respect to argument {}, which has multiplicity {}".format(self.arg(Q)),Qconst[Q])
		for Q in Qvar:
			print("Qvar contains the derivative with respect to argument {}, which has multiplicity {}".format(Q,Qvar[Q]))

class ProductExprEval(BinaryArithmeticOpEval):
	def __init__(self,L,R):
		super().__init__(L,R,ScalarShape())
		self.L = L 
		self.R = R 

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

	def printQ(self,d):
		[Qconst, Qvar] = self.buildQ(d)

		for Q in Qconst:
			print("Qconst contains the derivative with respect to argument {}, which has multiplicity {}".format(self.arg(Q)),Qconst[Q])
		for Q in Qvar:
			print("Qvar contains the derivative with respect to argument {}, which has multiplicity {}".format(Q,Qvar[Q]))


class QuotientExprEval(BinaryArithmeticOpEval):
	def __init__(self,L,R):
		super().__init__(L,R, L.shape())
		self.L = L 
		self.R = R 

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)
		F_keys = F.keys()

		Qconst = {}; Qvar = F 
		if d >= 2:
			keys_to_remove = []
			for Q in Qvar:
				num_zeros = Q.count(0)
				if num_zeros >= 2:
					keys_to_remove.append(Q)

			for key in keys_to_remove:
				del Qvar[key] 

		return Qconst, Qvar 

	def printQ(self,d):
		[Qconst, Qvar] = self.buildQ(d)

		for Q in Qconst:
			print("Qconst contains the derivative with respect to argument {}, which has multiplicity {}".format(self.arg(Q)),Qconst[Q])
		for Q in Qvar:
			print("Qvar contains the derivative with respect to argument {}, which has multiplicity {}".format(Q,Qvar[Q]))

class PowerExprEval(BinaryArithmeticOpEval):
	def __init__(self,L,R):
		super().__init__(L,R,L.shape())
		self.L = L 
		self.R = R 

	def buildQForOrder(self,d):
		F = self.buildFForOrder(d)

		Qconst = {}; Qvar = F 
		return Qconst, Qvar 

	def printQ(self,d):
		[Qconst, Qvar] = self.buildQ(d)

		for Q in Qconst:
			print("Qconst contains the derivative with respect to argument {}, which has multiplicity {}".format(self.arg(Q)),Qconst[Q])
		for Q in Qvar:
			print("Qvar contains the derivative with respect to argument {}, which has multiplicity {}".format(Q,Qvar[Q]))




