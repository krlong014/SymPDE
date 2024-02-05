from abc import ABC, abstractmethod 
from SymPDE.Expr import Expr
from SymPDE.ArithmeticExpr import UnaryExpr, BinaryExpr 
from SymPDE.UnaryExpr import UnivariateFuncExpr
from SymPDE.Coordinate import Coordinate 
from SymPDE.ExprShape import ScalarShape

class UnivariateFuncExprEval(UnivariateFuncExpr):
	def __init__(self,name,arg):
		super().__init__(expr, expr.shape())
		self._name = name 

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
