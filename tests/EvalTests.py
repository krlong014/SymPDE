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

	# #testing 2nd order on a coordinate unary minus
	# def test_A_coord_unary_2(self):
	# 	x = Coordinate(0)
	# 	g = -x
	# 	context = 'no context'

	# 	g_eval = g._makeEval(context)

	# 	[Aconst, Avar] = g_eval.buildAForOrder(2)

	# 	realAconst = {}
	# 	realAvar = {}

	# 	assert(Aconst == realAconst)
	# 	assert(Avar == realAvar)

	#testing 1st order on a coordinate sum
	def test_A_coord_sum_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x + y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x':1, 'y':1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	# #testing 2nd order on a coordinate sum
	# def test_A_coord_sum_2(self):
	# 	x = Coordinate(0)
	# 	y = Coordinate(1)
	# 	g = x + y 
	# 	context = 'no context'

	# 	g_eval = g._makeEval(context)

	# 	[Aconst, Avar] = g_eval.buildAForOrder(2)

	# 	realAconst = {}
	# 	realAvar = {}

	# 	assert(Aconst == realAconst)
	# 	assert(Avar == realAvar)

	#testing 1st order on a coordinate difference
	def test_A_coord_difference_1(self):
		x = Coordinate(0)
		y = Coordinate(1)
		g = x - y 
		context = 'no context'

		g_eval = g._makeEval(context)

		[Aconst, Avar] = g_eval.buildAForOrder(1)

		realAconst = {}
		realAvar = {'x':1, 'y':1}

		assert(Aconst == realAconst)
		assert(Avar == realAvar)

	# #testing 2nd order on a coordinate difference
	# def test_A_coord_difference_2(self):
	# 	x = Coordinate(0)
	# 	y = Coordinate(1)
	# 	g = x - y 
	# 	context = 'no context'

	# 	g_eval = g._makeEval(context)

	# 	[Aconst, Avar] = g_eval.buildAForOrder(2)

	# 	realAconst = {}
	# 	realAvar = {}

	# 	assert(Aconst == realAconst)
	# 	assert(Avar == realAvar)

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
	# # def test_A_coord_product_2(self):
	# 	x = Coordinate(0)
	# 	y = Coordinate(1)
	# 	g = x * y 
	# 	context = 'no context'

	# 	g_eval = g._makeEval(context)

	# 	[Aconst, Avar] = g_eval.buildAForOrder(2)

	# 	realAconst = {('x','y'): 2}
	# 	realAvar = {}

	# 	assert(Aconst == realAconst)
	# 	assert(Avar == realAvar)