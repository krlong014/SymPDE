from SymPDE.DerivSpecifier import DerivSpecifier, MultipleDeriv
from SymPDE.Coordinate import Coordinate
from SymPDE.HungryDiffOp import _Partial, _Gradient
from SymPDE.FunctionWithBasis import TestFunction, UnknownFunction
from SymPDE.Lagrange import Lagrange

if __name__=='__main__':
    
    
    bas = Lagrange(1)
    v = TestFunction(bas, 'v')
    u = UnknownFunction(bas, 'u')
    X = Coordinate(0, 'x')
    Y = Coordinate(1, 'y')

    I_v = DerivSpecifier(v)
    I_u = DerivSpecifier(u)
    dx_v = DerivSpecifier(v, _Partial(0))
    dy_v = DerivSpecifier(v, _Partial(1))
    grad_v = DerivSpecifier(v, _Gradient(3))
    grad_u = DerivSpecifier(u, _Gradient(3))
    x = DerivSpecifier(X)
    y = DerivSpecifier(Y)

    s = {I_v, I_u, dx_v, dy_v, grad_v, grad_u, x, y}

    #print(s)

    lam = MultipleDeriv([x,])
    mu = MultipleDeriv([I_u, dx_v, I_u])
    nu = MultipleDeriv([grad_v, I_v, x])
    print('lambda=',lam)
    print('mu=',mu)
    print('nu=',nu)

    m = {
        lam : 'C',
        mu : 'A',
        nu : 'B'
    }

    print('m=',m)