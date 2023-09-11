from . Expr import Expr
from . ArithmeticExpr import SumExpr, UnaryMinus, ProductExpr
from . FunctionWithBasis import FunctionWithBasis, TestFunction, UnknownFunction
from . DiffOp import DiffOpOnFunction, Partial
from . DerivSpecifier import DerivSpecifier
from . Lagrange import Lagrange
from . UnivariateFunc import Sin

from PyTab import Tab

def analyzeExpr(expr, verb=0):
    assert(isinstance(expr, Expr))

    tab = Tab()

    if verb>0:
        key = type(expr).__name__
        print(tab, 'Analyzing expr of type ', key)

    if isinstance(expr, SumExpr):
        return analyzeSum(expr, verb)
    if isinstance(expr, ProductExpr):
        return analyzeProduct(expr, verb)
    if isinstance(expr, UnaryMinus):
        return analyzeUnaryMinus(expr, verb)
    if isinstance(expr, FunctionWithBasis):
        return analyzeFunction(expr, verb)
    if isinstance(expr, DiffOpOnFunction):
        return analyzeDiffOp(expr, verb)

    return set()


def analyzeSum(expr, verb=0):

    if verb>0:
        tab = Tab()
        print(tab, 'analyzeSum:')

    assert(isinstance(expr, SumExpr))

    tab = Tab()
    tab1 = Tab()

    if verb>0:
        print(tab1, 'Left:')
    L = analyzeExpr(expr.left(), verb)
    if verb>0:
        print(tab1, 'Right:')
    R = analyzeExpr(expr.right(), verb)

    return L.union(R)


def analyzeUnaryMinus(expr, verb=0):

    assert(isinstance(expr, UnaryMinus))

    return analyzeExpr(expr.arg(), verb)


def analyzeFunction(expr, verb=0):

    assert(isinstance(expr, FunctionWithBasis))

    return {DerivSpecifier(expr)}


def analyzeDiffOp(expr, verb=0):

    assert(isinstance(expr, DiffOpOnFunction))

    op = expr.op()
    arg = expr.arg()

    return {DerivSpecifier(arg, op)}


if __name__=='__main__':

    v = TestFunction(Lagrange(1), 'v')
    dvdx = Partial(v, 0)
    u = UnknownFunction(Lagrange(1), 'u')
    dudx = Partial(u, 0)

    e1 = v + dvdx
    e2 = v + dvdx + 1
    e3 = v*v + dvdx
    e4 = Sin(v) + dvdx
    e5 = dvdx*dudx + 2*v

    def check(e):

        print('------------------------------------------------------')
        print('Checking expression ', e)

        derivs = analyzeExpr(e, verb=0)
        print('Functions and their derivatives: ', derivs)
        print(derivs)

        print('Does every term have a test function? Answer: ',
              e.everyTermHasTest())

        print('Is the expression linear and homogeneous in test functions?')
        print('Answer: ', e.isLinearInTests())

        print('Test functions are: ', e.getTests())
        print('Unknown functions are: ', e.getUnks())



    for e in (e1, e2, e3, e4, e5):
        check(e)
