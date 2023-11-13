import pytest
import itertools as it 
from random import randint
import numpy as np
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import UnaryMinus, SumExpr, ProductExpr, PowerExpr, QuotientExpr
import SymPDE.FunctionWithBasis as fwb
from SymPDE.BasisBase import BasisBase, ScalarBasisBase

class TestQSetConstruction:
	#testing 1st order on a coordinate unary minus
	def test_coord_unary_1(self):
		a0 = Coordinate(0)
		g = -a0

		[Qconst, Qvar] = g.buildQForOrder(1)

		realQconst = {0:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate unary minus
	def test_coord_unary_2(self):
		a0 = Coordinate(0)
		g = -a0

		[Qconst, Qvar] = g.buildQForOrder(2)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate unary minus
	def test_coord_unary_3(self):
		a0 = Coordinate(0)
		g = -a0

		[Qconst, Qvar] = g.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing up to 3rd order on a coordinate unary minus
	def test_coord_unary_up_to_3(self):
		a0 = Coordinate(0)
		g = -a0

		[Qconst, Qvar] = g.buildQUpToOrder(3)

		realQconst = {0:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate sum
	def test_coord_sum_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Qconst,Qvar] = g.buildQForOrder(1)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate sum
	def test_coord_sum_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Qconst,Qvar] = g.buildQForOrder(2)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate sum
	def test_coord_sum_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Qconst,Qvar] = g.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing up to 3rd order on a coordinate sum
	def test_coord_sum_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Qconst,Qvar] = g.buildQUpToOrder(3)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate difference
	def test_coord_difference_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Qconst, Qvar] = g.buildQForOrder(1)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate difference
	def test_coord_difference_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Qconst, Qvar] = g.buildQForOrder(2)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate difference
	def test_coord_difference_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Qconst, Qvar] = g.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing up to 3rd order on a coordinate difference
	def test_coord_difference_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Qconst, Qvar] = g.buildQUpToOrder(3)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate product
	def test_coord_product_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Qconst, Qvar] = g.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate product
	def test_coord_product_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Qconst, Qvar] = g.buildQForOrder(2)

		realQconst = {(0,1): 2}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate product
	def test_coord_product_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Qconst, Qvar] = g.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing up to 3rd order on a coordinate product
	def test_coord_product_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Qconst, Qvar] = g.buildQUpToOrder(3)

		realQconst = {(0,1): 2}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate quotient
	def test_coord_quotient_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Qconst, Qvar] = g.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate quotient
	def test_coord_quotient_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Qconst, Qvar] = g.buildQForOrder(2)

		realQconst = {}
		realQvar = {(0,1):2, (1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate quotient
	def test_coord_quotient_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Qconst, Qvar] = g.buildQForOrder(3)

		realQconst = {}
		realQvar = {(0,1,1):3, (1,1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing up to 3rd order on a coordinate quotient
	def test_coord_quotient_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Qconst, Qvar] = g.buildQUpToOrder(3)

		realQconst = {}
		realQvar = {0:1, 1:1, (0,1): 2, (1,1): 1, (0,1,1): 3, (1,1,1): 1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate power
	def test_coord_power_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Qconst, Qvar] = g.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate power
	def test_coord_power_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Qconst, Qvar] = g.buildQForOrder(2)

		realQconst = {}
		realQvar = {(0,0): 1, (0,1): 2, (1,1): 1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate power
	def test_coord_power_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Qconst, Qvar] = g.buildQForOrder(3)

		realQconst = {}
		realQvar = {(0,0,0): 1, (0,0,1): 3, (0,1,1): 3, (1,1,1): 1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing up to 3rd order on a coordinate power
	def test_coord_power_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Qconst, Qvar] = g.buildQUpToOrder(3)

		realQconst = {}
		realQvar = {0:1, 1:1, (0,0): 1, (0,1): 2, (1,1): 1, (0,0,0): 1, (0,0,1): 3, (0,1,1): 3, (1,1,1): 1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

class TestASetConstruction:
	#testing 1st order on a unary coordinate minus
	def test_coord_unary_1(self):
		a0 = Coordinate(0)
		g = -a0

		[Aconst, Avar] = g.buildAForOrder(1)

		realAconst = {0:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a unary coordinate minus
	def test_coord_unary_2(self):
		a0 = Coordinate(0)
		g = -a0

		[Aconst, Avar] = g.buildAForOrder(2)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a unary coordinate minus
	def test_coord_unary_3(self):
		a0 = Coordinate(0)
		g = -a0

		[Aconst, Avar] = g.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing up to 3rd order on a unary coordinate minus
	def test_coord_unary_up_to_3(self):
		a0 = Coordinate(0)
		g = -a0

		[Aconst, Avar] = g.buildAUpToOrder(3)

		realAconst = {0:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate sum
	def test_coord_sum_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Aconst, Avar] = g.buildAForOrder(1)

		realAconst = {0:1, 1:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate sum
	def test_coord_sum_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Aconst, Avar] = g.buildAForOrder(2)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate sum
	def test_coord_sum_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Aconst, Avar] = g.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing up to 3rd order on a coordinate sum
	def test_coord_sum_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		[Aconst, Avar] = g.buildAUpToOrder(3)

		realAconst = {0:1, 1:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate difference
	def test_coord_difference_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Aconst, Avar] = g.buildAForOrder(1)

		realAconst = {0:1, 1:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate difference
	def test_coord_difference_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Aconst, Avar] = g.buildAForOrder(2)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate difference
	def test_coord_difference_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Aconst, Avar] = g.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing up to 3rd order on a coordinate difference
	def test_coord_difference_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		[Aconst, Avar] = g.buildAUpToOrder(3)

		realAconst = {0:1, 1:1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate product
	def test_coord_product_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Aconst, Avar] = g.buildAForOrder(1)

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate product
	def test_coord_product_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Aconst, Avar] = g.buildAForOrder(2)

		realAconst = {(0,1): 2}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate product
	def test_coord_product_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Aconst, Avar] = g.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing up to 3rd order on a coordinate product
	def test_coord_product_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		[Aconst, Avar] = g.buildAUpToOrder(3)

		realAconst = {(0,1): 2}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate quotient
	def test_coord_quotient_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Aconst, Avar] = g.buildAForOrder(1)

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate quotient
	def test_coord_quotient_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Aconst, Avar] = g.buildAForOrder(2)

		realAconst = {}
		realAvar = {(0,1): 2, (1,1): 1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate quotient
	def test_coord_quotient_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Aconst, Avar] = g.buildAForOrder(3)

		realAconst = {}
		realAvar = {(0,1,1): 3, (1,1,1):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing up to 3rd order on a coordinate quotient
	def test_coord_quotient_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		[Aconst, Avar] = g.buildAUpToOrder(3)

		realAconst = {}
		realAvar = {0:1, 1:1, (0,1):2, (1,1):1, (0,1,1): 3, (1,1,1):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate power
	def test_coord_power_1(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Aconst, Avar] = g.buildAForOrder(1)

		realAconst = {}
		realAvar = {0:1, 1:1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate power
	def test_coord_power_2(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Aconst, Avar] = g.buildAForOrder(2)

		realAconst = {}
		realAvar = {(0,0):1, (0,1):2, (1,1):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate power
	def test_coord_power_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Aconst, Avar] = g.buildAForOrder(3)

		realAconst = {}
		realAvar = {(0,0,0):1, (0,0,1):3, (0,1,1):3, (1,1,1):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing up to 3rd order on a coordinate power
	def test_coord_power_up_to_3(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		[Aconst, Avar] = g.buildAUpToOrder(3)

		realAconst = {}
		realAvar = {0:1, 1:1, (0,0):1, (0,1):2, (1,1):1, (0,0,0):1, (0,0,1):3, (0,1,1):3, (1,1,1):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

class TestRSetConstruction:
	#testing on a linearized forward problem with coordinate unary minus built to hit
	def test_forward_coord_unary_minus_hit(self):
		a1 = Coordinate(1)
		g = -a1

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = [{1:1}]
		realRvar = []

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on a linearized forward problem with coordinate unary minus built to miss
	def test_forward_coord_unary_minus_miss(self):
		a0 = Coordinate(0)
		g = -a0

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = []
		realRvar = []

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on a linearized forward problem with coordinate sum
	def test_forward_coord_sum(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 + a1 

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = [{1:1}]
		realRvar = []

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on a linearized forward problem with coordinate difference
	def test_forward_coord_difference(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 - a1 

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = [{1:1}]
		realRvar = []

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on a linearized forward problem with coordinate product
	def test_forward_coord_product(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 * a1 

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = [{(0,1): 2}]
		realRvar = [{1:1}]

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on a linearized forward problem with coordinate quotient
	def test_forward_coord_quotient(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 / a1 

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = []
		realRvar = [{1:1}, {(0,1): 2}]

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on a linearized forward problem with coordinate power
	def test_forward_coord_power(self):
		a0 = Coordinate(0)
		a1 = Coordinate(1)
		g = a0 ** a1 

		P1 = {1:1}
		P2 = {(0,1): 1}
		Petitions = [P1, P2]

		[Rconst, Rvar] = g.buildR(Petitions)

		realRconst = []
		realRvar = [{1:1}, {(0,1): 2}]

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)