from graphviz import Digraph
from . Expr import (BinaryExpr, Coordinate, UnaryMinus, ConstantExprBase)
from . DiffOp import DiffOp, DiffOpOnFunction, Partial
import numpy.random as npr
from . UnivariateFunc import (UnivariateFuncExpr, Exp, Sqrt, Log, Cos, Sin, Tan,
                Cosh, Sinh, Tanh, ArcCos, ArcSin, ArcTan,
                ArcCosh, ArcSinh, ArcTanh)
from . FunctionWithBasis import UnknownFunction, TestFunction, FunctionWithBasis
from . BasisBase import ScalarBasisBase

def vizExpr(expr, filename):

    g = Digraph('g', filename=filename,
        node_attr={'fontsize':'10', 'width' : '0.25', 'height' : '0.15',
            'fontname' : 'times:italic'},
            edge_attr={'dir' : 'back'})

    g.node('root')

    graphExpr(g, 'root', expr)

    g.view()

def makeID():
    return str(npr.randint(0,2**63))



def graphExpr(g, parent, expr):

    if isinstance(expr, BinaryExpr):
        graphBinaryOp(g, parent, expr)

    elif isinstance(expr, UnivariateFuncExpr):
        graphUnivariateFunc(g, parent, expr)

    elif isinstance(expr, UnaryMinus):
         graphUnaryMinus(g, parent, expr)

    elif isinstance(expr, Coordinate):
        graphCoordinate(g, parent, expr)

    elif isinstance(expr, DiffOpOnFunction):
        graphDiffOpOnFunction(g, parent, expr)

    elif isinstance(expr, DiffOp):
        graphDiffOp(g, parent, expr)

    elif isinstance(expr, FunctionWithBasis):
        graphFunctionWithBasis(g, parent, expr)

    elif isinstance(expr, ConstantExprBase):
        graphConstant(g, parent, expr)

    else:
        raise ValueError('invalid input {}'.format(expr))


def graphBinaryOp(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.opString(), shape='circle', fontsize='16')
    g.edge(parent, myID)

    graphExpr(g, myID, expr.left())
    graphExpr(g, myID, expr.right())

def graphUnaryMinus(g, parent, expr):

    myID = makeID()

    g.node(myID, label='-', shape='circle')
    g.edge(parent, myID)

    graphExpr(g, myID, expr.arg())

def graphUnivariateFunc(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.name(), shape='box')
    g.edge(parent, myID)

    graphExpr(g, myID, expr.arg())


def graphCoordinate(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.__str__(), shape='box')
    g.edge(parent, myID)


def graphDiffOp(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.op().__str__(), shape='ellipse')
    g.edge(parent, myID)

    graphExpr(g, myID, expr.arg())

def graphDiffOpOnFunction(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.__str__(), shape='octagon')
    g.edge(parent, myID)

def graphFunctionWithBasis(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.__str__(), shape='octagon')
    g.edge(parent, myID)


def graphConstant(g, parent, expr):

    myID = makeID()

    g.node(myID, label=expr.__str__(), shape='diamond')
    g.edge(parent, myID)



if __name__=='__main__':

    x = Coordinate(0)
    y = Coordinate(1)
    bas = ScalarBasisBase(1)
    u = UnknownFunction(bas, 'u')

    f = -x + y*Partial(Cos(y), y) + u - Partial(u, x) + Exp(-2*u)

    vizExpr(f, 'expr.gv')
