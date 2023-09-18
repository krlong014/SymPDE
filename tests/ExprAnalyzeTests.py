import pytest
import itertools as it 
from random import randint
import numpy as np
from ExprEval import buildAllQ
from ExprEval import refineQsets
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import SumExpr, ProductExpr, PowerExpr, QuotientExpr

#all tests regarding Qset construction
class TestQSetConstruction:
	#makes sure assertion error raised with no arguments
	def test_NoArgs(self):
		pytest.raises(AssertionError, buildAllQ, randint(1,9), 0)

	#double checking 1st order on 1 arg expr
	def test_1_1(self):
		Qsets = buildAllQ(1,1)
		assert(Qsets == [[1]])

	#double checking 1st order on n arg expr
	def test_1_n(self):
		n = randint(3,9)
		Qsets = buildAllQ(1,n)
		realQsets = [i+1 for i in range(n)]
		realQsets = [realQsets]
		assert(realQsets == Qsets)

	#doule checking 3rd order on 1 arg expr
	def test_3_1(self):
		Qsets = buildAllQ(3,1)
		realQsets = [[1], [(1,1)], [(1,1,1)]]
		assert(Qsets == realQsets)

	#double checking 3rd order on 3 arg expr
	def test_3_3(self):
		Qsets = buildAllQ(3,3)
		realQsets = [[1,2,3], [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)], [(1,1,1), (1,1,2), (1,1,3), (1,2,1), (1,2,2), (1,2,3), (1,3,1), (1,3,2), (1,3,3), (2,1,1), (2,1,2), (2,1,3), (2,2,1), (2,2,2), (2,2,3), (2,3,1), (2,3,2), (2,3,3), (3,1,1), (3,1,2), (3,1,3), (3,2,1), (3,2,2), (3,2,3), (3,3,1), (3,3,2), (3,3,3)]]
		assert(Qsets == realQsets)


#all tests regarding Qset partitioning
class TestQSetRefinement:
	#double checking sum expr refinement on 3rd order, 2 arg sum
	def test_SumRefine(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = SumExpr(a1,a2,1)

		Qsets = buildAllQ(3,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = [1,2]
		realQvar = [[(1,1), (1,2), (2,1), (2,2)], [(1,1,1),(1,1,2),(1,2,1),(1,2,2),(2,1,1),(2,1,2),(2,2,1),(2,2,2)]]
		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking diff expr refinement on 4th order, 2 arg difference
	def test_DiffRefine(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = SumExpr(a1,a2,-1)

		Qsets = buildAllQ(4,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = [1,2]
		realQvar = [[(1,1), (1,2), (2,1), (2,2)], [(1,1,1),(1,1,2),(1,2,1),(1,2,2),(2,1,1),(2,1,2),(2,2,1),(2,2,2)], [(1,1,1,1),(1,1,1,2),(1,1,2,1),(1,1,2,2),(1,2,1,1),(1,2,1,2),(1,2,2,1),(1,2,2,2),(2,1,1,1),(2,1,1,2),(2,1,2,1),(2,1,2,2),(2,2,1,1),(2,2,1,2),(2,2,2,1),(2,2,2,2)]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking product expr refinement on 3rd order, 2 arg product
	def test_ProdRefine_3_2(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = ProductExpr(a1,a2)

		Qsets = buildAllQ(3,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = [(1,1), (1,2), (2,1), (2,2)]
		realQvar = [[1,2],[(1,1,1),(1,1,2),(1,2,1),(1,2,2),(2,1,1),(2,1,2),(2,2,1),(2,2,2)]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking product expr refinement on 1st order, 2 arg product
	def test_ProdRefine_1_2(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = ProductExpr(a1,a2)

		Qsets = buildAllQ(1,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = []
		realQvar = [[1,2]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking quotient expr refinement on 3rd order, 2 arg product
	def test_QuotRefine_3_2(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = QuotientExpr(a1,a2)

		Qsets = buildAllQ(3,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = [(1,1), (1,2), (2,1), (2,2)]
		realQvar = [[1,2],[(1,1,1),(1,1,2),(1,2,1),(1,2,2),(2,1,1),(2,1,2),(2,2,1),(2,2,2)]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking quotient expr refinement on 1st order, 2 arg product
	def test_QuotRefine_1_2(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = QuotientExpr(a1,a2)

		Qsets = buildAllQ(1,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = []
		realQvar = [[1,2]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking power expr refinement on 3rd order, 2 arg product
	def test_PowRefine_3_2(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = PowerExpr(a1,a2)

		Qsets = buildAllQ(3,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = [(1,1), (1,2), (2,1), (2,2)]
		realQvar = [[1,2],[(1,1,1),(1,1,2),(1,2,1),(1,2,2),(2,1,1),(2,1,2),(2,2,1),(2,2,2)]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#double checking product expr refinement on 1st order, 2 arg product
	def test_PowRefine_1_2(self):
		a1 = Coordinate(0)
		a2 = Coordinate(1)
		g = PowerExpr(a1,a2)

		Qsets = buildAllQ(1,2)
		[Qconst, Qvar] = refineQsets(Qsets, g)

		realQconst = []
		realQvar = [[1,2]]

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

#rename to ExprAnalyzeTest (and ExprAnalyze)