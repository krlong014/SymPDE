from graphviz import Digraph
from . Coordinate import Coordinate
from . ConstantExpr import ConstantExprBase
from . ArithmeticExpr import BinaryExpr, UnaryMinus
from . DiffOp import DiffOp, DiffOpOnFunction, Partial
import numpy.random as npr
from . UnivariateFunc import (UnivariateFuncExpr, Exp, Sqrt, Log, Cos, Sin, Tan,
                Cosh, Sinh, Tanh, ArcCos, ArcSin, ArcTan,
                ArcCosh, ArcSinh, ArcTanh)
from . FunctionWithBasis import UnknownFunction, TestFunction, FunctionWithBasis
from . Lagrange import Lagrange

def vizExpr(idToFuncMap, funcToIDMap, expr, filename):

    g = Digraph('g', filename=filename,
        node_attr={'fontsize':'10', 'width' : '0.25', 'height' : '0.15',
            'fontname' : 'times:italic'},
            edge_attr={'dir' : 'back'})

    g.node('root', label='f={}'.format(str(expr)), shape='box')

    graphExpr(g, idToFuncMap, funcToIDMap, 'root', expr)

    g.view()


class NodeCounter:
    nodeID = 0

    @classmethod
    def nextID(cls):
        tmp = cls.nodeID
        cls.nodeID += 1
        return tmp


def makeID():
    return str(NodeCounter.nextID())
    #return str(npr.randint(0,2**63))

def tag(myID, eString):
    rtn = 'E{}: {}'.format(myID, eString)
    return rtn



def graphExpr(g, idToFuncMap, funcToIDMap, parent, expr):

    if isinstance(expr, BinaryExpr):
        graphBinaryOp(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, UnivariateFuncExpr):
        graphUnivariateFunc(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, UnaryMinus):
         graphUnaryMinus(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, Coordinate):
        graphCoordinate(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, DiffOpOnFunction):
        graphDiffOpOnFunction(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, DiffOp):
        graphDiffOp(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, FunctionWithBasis):
        graphFunctionWithBasis(g, idToFuncMap, funcToIDMap, parent, expr)

    elif isinstance(expr, ConstantExprBase):
        graphConstant(g, idToFuncMap, funcToIDMap, parent, expr)

    else:
        raise ValueError('invalid input {}'.format(expr))


def graphBinaryOp(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, label=tag(myID,expr.opString()), shape='circle')
    g.edge(parent, myID)

    graphExpr(g, idToFuncMap, funcToIDMap, myID, expr.left())
    graphExpr(g, idToFuncMap, funcToIDMap, myID, expr.right())

def graphUnaryMinus(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, label=tag(myID, '-'), shape='circle')
    g.edge(parent, myID)

    graphExpr(g, idToFuncMap, funcToIDMap, myID, expr.arg())

def graphUnivariateFunc(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, label=tag(myID,expr.name()), shape='box')
    g.edge(parent, myID)

    graphExpr(g, idToFuncMap, funcToIDMap, myID, expr.arg())


def graphCoordinate(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, tag(myID,expr.__str__()), shape='box')
    g.edge(parent, myID)


def graphDiffOp(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, tag(myID,expr.op().__str__()),
           shape='ellipse')
    g.edge(parent, myID)

    graphExpr(g, idToFuncMap, funcToIDMap, myID, expr.arg())

def graphDiffOpOnFunction(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, label=tag(myID,expr.__str__()), shape='octagon')
    g.edge(parent, myID)

    graphFunctionWithBasis(g, idToFuncMap, funcToIDMap, myID, expr.arg())

def graphFunctionWithBasis(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = funcToIDMap[expr]

    g.node(myID, label=tag(myID, expr.__str__()), shape='circle')
    g.edge(parent, myID)


def graphConstant(g, idToFuncMap, funcToIDMap, parent, expr):

    myID = makeID()

    g.node(myID, label=tag(myID,expr.__str__()), shape='diamond')
    g.edge(parent, myID)



if __name__=='__main__':

    x = Coordinate(0)
    y = Coordinate(1)
    bas = Lagrange(1)
    v = UnknownFunction(bas, 'v')
    u = UnknownFunction(bas, 'u')

    vID = makeID()
    uID = makeID()

    idToFuncMap = {
        vID : v,
        uID : u
    }

    funcToIDMap = {
        v : vID,
        u : uID
    }


    #f = Partial(v,x)*Partial(u,x) + v*(x + y*Partial(Cos(y), y) + u + Exp(-2*u))
    f = v*u + Partial(v,x)*Sin(u) + Partial(v,x)*Partial(u,x)
    print(str(f))

    vizExpr(idToFuncMap, funcToIDMap, f, 'expr.gv')
