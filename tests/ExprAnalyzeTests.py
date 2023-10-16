import pytest
import itertools as it 
from random import randint
import numpy as np
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import UnaryMinus, SumExpr, ProductExpr, PowerExpr, QuotientExpr

class TestQSets:
	#testing 1st order on a coordinate unary minus
	def test_unary_1(self):
		a0 = Coordinate(0)
		g = -a0

		[Qconst, Qvar] = g.buildQ(1)

		realQconst = {0: 1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordiante unary minus
	def test_unary_2(self):
		a0 = Coordinate(0)
		g = -a0

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {0: 1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a 2D coordinate sum
	def test_sum_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Qconst, Qvar] = g.buildQ(1)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar) 

	#testing 2nd order on a 2D coordinate sum
	def test_sum_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a 2D coordinate difference
	def test_diff_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a 2D coordinate difference
	def test_diff_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a 2D coordinate product
	def test_prod_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0*a1 

		[Qconst, Qvar] = g.buildQ(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a 2D coordinate product
	def test_prod_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0*a1 

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {(0,1):2}
		realQvar = {0:1,1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a 2D coordinate product
	def test_prod_3(self):
		self.test_prod_2()
		# a0 = Coordinate(0)
		# a1 = Coordinate(1)
		# g = a0*a1 

		# [Qconst, Qvar] = g.buildQ(3)

		# realQconst = {(0,1):2}
		# realQvar = {0:1,1:1}

		# assert(Qconst == realQconst)
		# assert(Qvar == realQvar)

	#testing 1st order on a 2D coordinate quotient
	def test_quot_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0/a1 

		[Qconst, Qvar] = g.buildQ(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a 2D coordinate quotient
	def test_quot_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0/a1 

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {}
		realQvar = {0:1,1:1,(0,1):2,(1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a 2D coordinate quotient
	def test_quot_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0/a1 

		[Qconst, Qvar] = g.buildQ(3)

		realQconst = {}
		realQvar = {0:1, 1:1, (0,1):2, (1,1): 1, (0,1,1):3, (1,1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a 2D coordinate power
	def test_power_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0**a1 

		[Qconst, Qvar] = g.buildQ(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a 2D coordinate power
	def test_power_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0**a1 

		[Qconst, Qvar] = g.buildQ(2)

		realQconst = {}
		realQvar = {0:1, 1:1, (0,0):1, (0,1):2, (1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a 2D coordinate power
	def test_power_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0**a1 

		[Qconst, Qvar] = g.buildQ(3)

		realQconst = {}
		realQvar = {0:1, 1:1, (0,0):1, (0,1):2, (1,1):1, (0,0,0):1, (0,0,1):3, (0,1,1):3, (1,1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

class TestASets:
	#testing 1st order on a unary coordinate minus
	def test_unary_1(self):
		a0 = Coordinate(0)
		g = -a0

		[Aconst, Avar] = g.buildA(1)

		realAconst = {0:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a unary coordinate minus
	def test_unary_2(self):
		a0 = Coordinate(0)
		g = -a0

		[Aconst, Avar] = g.buildA(2)

		realAconst = {0:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing on a 2D sum of coordinates
	def test_sum(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Aconst, Avar] = g.buildA()

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing on a 2D coordinate differnce
	def test_diff(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Aconst, Avar] = g.buildA()

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing on a 2D coordinate product
	def test_prod(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Aconst, Avar] = g.buildA()

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing on a 2D coordinate quotient
	def test_quot(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Aconst, Avar] = g.buildA()

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing on a 2D coordinate power
	def test_power(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Aconst, Avar] = g.buildA()

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

##what about Q^C conditions for A set construction

##make a function that prints out a "human-readable Q" so I know what I'm reading
##a few months from now