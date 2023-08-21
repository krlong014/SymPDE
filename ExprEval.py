import itertools as it

def part(n):
	if n == 1:
		return [1]

	parts = [] 

	for i in range(n):
		for j in range(len(part(i))):
			parts.append([n - i] + [part(i)[j]])

	return parts

def intPart(n):
	return [[n]] + part(n)

# testParts = intPart(4)
# print("partitions = ",testParts)

#assuming order d and n arguments
def buildQ(d,n):
	dummyIter = [(i+1) for i in range(n)]

	Q = list(it.product(dummyIter,repeat=d))
	return Q

Q = buildQ(2,3)


class Qset():
	def __init__(self, ordercap, ExprWithChildrenInstance):
		self.ordercap = ordercap
		self.ExperWithChildrenInstance = ExprWithChildrenInstance

		def buildSpecific(self, order):
			#n = number of arguments in ExprWithChildrenInstance
			dummyIter = [(i + 1) for i in range(n)]
			Qfull = list(it.product(dummyIter, repeat=order))
			return Qfull

		def buildAll(self):
			#n = number of arguments in ExprWithChildrenInstance
			Qsets = [] * self.ordercap
			for i in range(self.ordercap):
				#not sure what goes before buildSpecific
				Qsets[i] = buildSpecific(i)

			return Qsets


		def refine(self):
			if isinstance(self.ExprWithChildrenInstance, SumExpr):
				if self.order == 1:
					Qconst = Qfull
					Qvar = None 
				else:
					Qconst = None 
					Qvar = Qfull

			if isinstance(self.ExprWithChildrenInstance, ProductExpr) 
			or isinstance(self.ExprWithChildrenInstance, QuotientExpr) 
			or isinstance(self.ExprWithChildrenInstance, PowerExpr):
				if self.order == 2:
					Qconst = Qfull
					Qvar = None 
				else:
					Qconst = None
					Qvar = Qfull 

			return [Qconst, Qvar]

		def refineAll(self):
			Qsets = self.buildAll()
			QconstSets = [] * self.ordercap 
			QvarSets = [] * self.ordercap

			for i in range(self.ordercap):
				[QconstSets[i], QvarSets[i]] = Qsets[i].refine()

			return [QconstSets, QvarSets]



# print("Q = ",Q)

#The constant Q sets will depend on the type of Expr
	#sum/diff expr: Q_1^C = Q_1; Q_i^V = Q_i for all i != 1
	#prod/quot/exp expr: Q_2^C = Q_2; Q_i^V = Q_i fpr all i != 2