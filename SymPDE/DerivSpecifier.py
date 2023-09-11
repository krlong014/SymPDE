from functools import total_ordering
from . DiffOp import HungryDiffOp, _Partial, _Gradient, _IdentityOp
from . FunctionWithBasis import FunctionWithBasis, UnknownFunction, TestFunction
from . Lagrange import Lagrange
from . OrderedTuple import OrderedTuple



@total_ordering
class DerivSpecifier:

    def __init__(self, func, op=_IdentityOp()):
        assert(isinstance(op, HungryDiffOp))

        assert(isinstance(func, FunctionWithBasis))
        assert(func.isTest() or func.isUnknown())

        self.stuff = (func, op)

    def __eq__(self, other):
        assert(isinstance(other, DerivSpecifier))
        return self.op()==other.op() and self.funcID() == other.funcID()

    def __le__(self, other):
        assert(isinstance(other, DerivSpecifier))
        if self.op() < other.op():
            return True
        if self.op() > other.op():
            return False
        if self.funcID() < other.funcID():
            return True
        return False

    def func(self):
        return self.stuff[0]

    def funcID(self):
        return self.func().funcID()

    def op(self):
        return self.stuff[1]

    def __hash__(self):
        return hash(self.stuff)

    def __repr__(self):
        return 'DerivSpecifier(func={}, op={})'.format(self.stuff[0], self.stuff[1])

    def __str__(self):
        if self.op()==_IdentityOp():
            return str(self.func())
        else:
            return '{}({})'.format(self.op(), self.func())


class MultipleDeriv(OrderedTuple):
    def __init__(self, derivs):
        tmp = []
        for d in derivs:
            tmp.append(d)
        super().__init__(tmp)

    def order(self):
        return self.__len__()

    def __repr__(self):
        return 'MultipleDeriv{}'.format(OrderedTuple.__str__(self))

    def __str__(self):
        return 'MultipleDeriv{}'.format(OrderedTuple.__str__(self))











if __name__=='__main__':
    bas = Lagrange(1)
    v = TestFunction(bas, 'v')
    u = TestFunction(bas, 'u')

    I_v = DerivSpecifier(v)
    I_u = DerivSpecifier(u)
    dx_v = DerivSpecifier(v, _Partial(0))
    dy_v = DerivSpecifier(v, _Partial(1))
    grad_v = DerivSpecifier(v, _Gradient(3))
    grad_u = DerivSpecifier(u, _Gradient(3))

    s = {I_v, I_u, dx_v, dy_v, grad_v, grad_u}

    print(s)

    mu = MultipleDeriv([I_u, dx_v, I_u])
    nu = MultipleDeriv([grad_v, I_v])
    print('mu=',mu)
    print('nu=',nu)

    m = {
        mu : 'A',
        nu : 'B'
    }

    print('m=',m)
