import itertools as it
from abc import ABC, abstractmethod
from collections.abc import Iterable
from SymPDE.ExprShape import ExprShape
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import SumExpr, ProductExpr, PowerExpr, QuotientExpr
import numpy as np
from scipy.special import binom
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
		return [0]

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
def buildQ(d,n,multiplicities=False):
	if d == 1:
		Q = [i for i in range(n)]
	else:
		dummyIter = [i for i in range(n)]

		Q = list(it.combinations_with_replacement(dummyIter,d))

	mults = [[int(binom(d,i)) for i in range(len(Q))]]

	Q = Q + mults 
	return Q


## builds all Q sets with the highest order being ordercap for an n argument expr
##this needs to be moved into ExprWithChildren
def buildAllQ(ordercap,n):
	assert(n >= 1)

	Qsets = []
	for i in range(ordercap):
		Qsets.append(buildQ(i+1,n))

	return Qsets
	


# Qsets = buildAllQ(3,2)
# print("Qsets = ",Qsets)


# a1 = Coordinate(0)
# a2 = Coordinate(1)
# g = ProductExpr(a1,a2)

# Qsets = g.buildAllQUpToOrder(3)
# print("Qsets = ",Qsets)
# [Qconst, Qvar] = g.refineQ(Qsets)
# print("Qconst = {}, Qvar = {}".format(Qconst,Qvar))

# Q = g.buildAllQUpToOrder(3)
# print("Q ")

# A = a2.buildAllAUpToOrder(1)
# print("A = ",A)

# Qsets = g.buildAllQUpToOrder(3)
# print("Qsets = ",Qsets)
# [Qconst, Qvar] = g.refineQ(Qsets)
# print("Qconst = {}, Qvar = {}".format(Qconst, Qvar))

# #these "picees" can each be moved into their respective subclasses of ArithmeticExpr
# #maybe even separate by Linear/Nonlinear
# def refineQsets(Qsets, g):
# 	order = len(Qsets)

# 	if isinstance(g,SumExpr):
# 		Qconst = Qsets.pop(0)
# 		Qvar = []

# 	if isinstance(g,ProductExpr) or isinstance(g, QuotientExpr) or isinstance(g,PowerExpr):
# 		if order >= 2:
# 			Qvar = Qsets.pop(0)
# 			Qconst = Qsets.pop(0)

# 			for Q in Qconst:
# 				if Q[0] == Q[1]:
# 					Qconst.remove(Q)

# 			# print("remove_indeces = ",remove_indeces)
# 		else:
# 			Qconst = []
# 			Qvar = Qsets
	
# 	return(Qconst, Qvar)

