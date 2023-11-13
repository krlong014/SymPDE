from collections.abc import Iterable
from SymPDE.ExprShape import ExprShape
from SymPDE.Coordinate import Coordinate
from SymPDE.ArithmeticExpr import SumExpr, ProductExpr, PowerExpr, QuotientExpr
import SymPDE.FunctionWithBasis as fwb
from SymPDE.BasisBase import BasisBase, ScalarBasisBase

a0 = Coordinate(0) #u
a1 = Coordinate(1) #v
g = a0 * a1

P1 = {1:1}
P2 = {(0,1):2}
Petitions = [P1, P2]
[Rconst, Rvar] = g.buildR(Petitions)
print("Rconst = {}, Rvar = {}".format(Rconst,Rvar))

##finish d = 3 case 
##rebuild tests
##buildR command