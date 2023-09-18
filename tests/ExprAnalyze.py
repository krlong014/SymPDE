import itertools as it
from abc import ABC, abstractmethod
from collections.abc import Iterable
from SymPDE.ExprShape import ExprShape
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import SumExpr, ProductExpr, PowerExpr, QuotientExpr
import numpy as np
#SumExpr
#ProductExpr
#QuotientExpr
#PowerExpr


## flattens a list of lists into a single list
def flatten(lst):
	# print("triggered flatten({})".format(lst))
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



## intermediate step in intPart
def part(n):
	# print("triggered part({})".format(n))
	if n == 1:
		return [1]

	parts = [] 

	for i in range(n):
		for j in range(len(part(i))):
			parts.append(flatten([n - i] + [part(i)[j]]))

	return parts

## finds all integer patitions of an integer n
def intPart(n):
	# print("triggered intPart({})".format(n))
	return [[n]] + part(n)


## assuming order d and n arguments, builds the Q set
def buildQ(d,n):
	if d == 1:
		Q = [i+1 for i in range(n)]
	else:
		dummyIter = [(i+1) for i in range(n)]

		Q = list(it.product(dummyIter,repeat=d))
	
	return Q


## builds all Q sets with the highest order being ordercap for an n argument expr
def buildAllQ(ordercap,n):
	assert(n >= 1)

	Qsets = []
	for i in range(ordercap):
		Qsets.append(buildQ(i+1,n))

	return Qsets

Qsets = buildAllQ(3,3)
print("Qsets = ",Qsets)


# a1 = Coordinate(0)
# a2 = Coordinate(1)
# g = PowerExpr(a1,a2)
# print("g = ",g)

def refineQsets(Qsets, g):
	order = len(Qsets)

	if isinstance(g,SumExpr):
		Qconst = Qsets.pop(0)
		Qvar = Qsets

	if isinstance(g,ProductExpr) or isinstance(g, QuotientExpr) or isinstance(g,PowerExpr):
		if order >= 2:
			Qconst = Qsets.pop(1)
			Qvar = Qsets 
		else:
			Qconst = []
			Qvar = Qsets 
	
	return(Qconst, Qvar)


# [Qconst, Qvar] = refineQsets(Qsets,g)
# print("Qconst = {}, Qvar = {}".format(Qconst,Qvar))

#double check that all Q sets are being calculated correctly
#make an ExprEvalTests using pytest
	#classes should start with Test....
	#methods should start with test_....
	#each test has condition true/false 
#start by finding out how to get more info out of pytest
