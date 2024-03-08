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
	def test_Q_coord_unary_1(self):
		x = Coordinate(0)
		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)

		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {0:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate unary minus
	def test_Q_coord_unary_2(self):
		x = Coordinate(0)
		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)

		[Qconst, Qvar] = g_eval.buildQForOrder(2)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate unary minus
	def test_Q_coord_unary_3(self):
		x = Coordinate(0)
		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)

		[Qconst, Qvar] = g_eval.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate sum
	def test_Q_coord_sum_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate sum
	def test_Q_coord_sum_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(2)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate sum
	def test_Q_coord_sum_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate difference
	def test_Q_coord_difference_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate difference
	def test_Q_coord_difference_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(2)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate difference
	def test_Q_coord_difference_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate product
	def test_Q_coord_product_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x*y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate product
	def test_Q_coord_product_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x*y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(2)

		realQconst = {(0,1): 2}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate product
	def test_Q_coord_product_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x*y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(3)

		realQconst = {}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate quotient
	def test_Q_coord_quotient_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x / y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate quotient
	def test_Q_coord_quotient_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x / y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(2)

		realQconst = {}
		realQvar = {(0,1): 2, (1,1): 1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate quotient
	def test_Q_coord_quotient_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x / y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(3)

		realQconst = {}
		realQvar = {(0,1,1):3, (1,1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a coordinate power
	def test_Q_coord_power_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x ** y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a coordinate power
	def test_Q_coord_power_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x ** y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(2)

		realQconst = {}
		realQvar = {(0,0):1, (0,1):2, (1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 3rd order on a coordinate power
	def test_Q_coord_power_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x ** y
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(3)

		realQconst = {}
		realQvar = {(0,0,0):1, (0,0,1):3, (0,1,1):3, (1,1,1):1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 1st order on a simple, coordinate mixed example
	def test_Q_coord_mixed1_1(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x + (x*y)
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {0:1, 1:1}
		realQvar = {}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

	#testing 2nd order on a simple, coordinate, mixed ex
	def test_Q_coord_mixed2_1(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x ** (y - (3/x))
		context = 'no context'

		g_eval = g._makeEval(context)
		[Qconst, Qvar] = g_eval.buildQForOrder(1)

		realQconst = {}
		realQvar = {0:1, 1:1}

		assert(Qconst == realQconst)
		assert(Qvar == realQvar)

class TestASetConstruction:
	#testing 1st order on a coordinate unary minus
	def test_A_coord_unary_1(self):
		x = Coordinate(0)
		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {'x': 1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate unary minus
	def test_A_coord_unary_2(self):
		x = Coordinate(0)
		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate unary minus
	def test_A_coord_unary_3(self):
		x = Coordinate(0)
		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate sum
	def test_A_coord_sum_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {'x':1, 'y':1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate sum
	def test_A_coord_sum_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate sum
	def test_A_coord_sum_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate difference
	def test_A_coord_difference_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {'x':1, 'y':1}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate difference
	def test_A_coord_difference_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate difference
	def test_A_coord_difference_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate product
	def test_A_coord_product_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x * y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x': 1, 'y': 1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)


	#testing 2nd order on a coordinate product
	def test_A_coord_product_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x * y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {('x','y'): 2}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate product
	def test_A_coord_product_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x * y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(3)

		realAconst = {}
		realAvar = {}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordinate quotient
	def test_A_coord_quotient_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x / y 
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x': 1, 'y': 1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordinate quotient
	def test_A_coord_quotient_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x / y 
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {}
		realAvar = {('x','y'): 2, ('y','y'): 1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordinate quotient
	def test_A_coord_quotient_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x / y 
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(3)

		realAconst = {}
		realAvar = {('x','y','y'):3, ('y','y','y'):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a coordiante power
	def test_A_coord_power_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x ** y 
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x': 1, 'y': 1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 2nd order on a coordiante power
	def test_A_coord_power_2(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x ** y 
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {}
		realAvar = {('x','x'): 1, ('x','y'):2, ('y','y'):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 3rd order on a coordiante power
	def test_A_coord_power_3(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x ** y 
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(3)

		realAconst = {}
		realAvar = {('x','x','x'):1, ('x','x','y'):3, ('x','y','y'):3, ('y','y','y'):1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	#testing 1st order on a simple, coordinate, mixed expr
	def test_A_coord_mixed1_1(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x + (x*y)
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x':1, 'y': 1}

	#testing 2nd order on a simple, coordinate, mixed expr
	def test_A_coord_mixed1_2(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x + (x*y)
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(2)

		realAconst = {('x','y'):2}
		realAvar = {}
	

	#testing 1st order on a simple, coordinate, mixed ex
	def test_A_coord_mixed2_1(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x ** (y - (y/x))
		context = 'no context'

		g_eval = g._makeEval(context)
		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x':1, 'y':1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

class TestRSetConstruction:
	#testing on coordinate unary minus
	def test_R_coord_unary_minus(self):
		x = Coordinate(0)

		g = -x
		context = 'no context'

		g_eval = g._makeEval(context)
		Petitions = ['x', 'y', ('x','y')]
		[Rconst, Rvar] = g_eval.buildR(Petitions)

		realRconst = {'x':1}
		realRvar = {}

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on coordinate sum
	def test_R_coord_sum(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x + y
		context = 'no context'

		g_eval = g._makeEval(context)
		Petitions = ['x', 'y', ('x','y')]
		[Rconst, Rvar] = g_eval.buildR(Petitions)

		realRconst = {'x':1, 'y':1}
		realRvar = {}

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on coordinate product
	def test_R_coord_product(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x * y
		context = 'no context'

		g_eval = g._makeEval(context)
		Petitions = ['x', 'y', ('x','y')]
		[Rconst, Rvar] = g_eval.buildR(Petitions)

		realRconst = {('x','y'): 2}
		realRvar = {'x':1, 'y':1}

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)


	#testing on coordinate quotient
	def test_R_coord_quotient(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x / y
		context = 'no context'

		g_eval = g._makeEval(context)
		Petitions = ['x', 'y', ('x','y')]
		[Rconst, Rvar] = g_eval.buildR(Petitions)

		realRconst = {}
		realRvar = {'x':1, 'y':1, ('x','y'):2}

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	#testing on coordinate power
	def test_R_coord_power(self):
		x = Coordinate(0)
		y = Coordinate(1)

		g = x ** y
		context = 'no context'

		g_eval = g._makeEval(context)
		Petitions = ['x', 'y', ('x','y')]
		[Rconst, Rvar] = g_eval.buildR(Petitions)

		realRconst = {}
		realRvar = {'x':1, 'y':1, ('x','y'):2}

		assert(Rconst == realRconst)
		assert(Rvar == realRvar)

	# #testing on coordinate mixed problem
	# def test_R_coord_mixed1(self):
	# 	x = Coordinate(0)
	# 	y = Coordinate(1)

	# 	g = x + (x*y)
	# 	context = 'no context'

	# 	g_eval = g._makeEval(context)
	# 	Petitions = ['x', 'y', ('x','y')]
	# 	[Rconst, Rvar] = g_eval.buildR(Petitions)

	# 	realRconst = {('x','y'):2}
	# 	realRvar = {'x':1, 'y': 1}

	# 	assert(Rconst == realRconst)
	# 	assert(Rvar == realRvar)



