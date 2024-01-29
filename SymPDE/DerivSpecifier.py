from functools import total_ordering
from . DiffOp import HungryDiffOp, _Partial, _Gradient, _IdentityOp
from . FunctionWithBasis import FunctionWithBasis, UnknownFunction, TestFunction
from . Lagrange import Lagrange
from . OrderedTuple import OrderedTuple
from . Coordinate import Coordinate



@total_ordering
class DerivSpecifier:

    def __init__(self, funcOrCoord, op=_IdentityOp()):

      assert(isinstance(op, HungryDiffOp))

      assert(isinstance(funcOrCoord, FunctionWithBasis) 
              or isinstance(funcOrCoord, Coordinate))

      if isinstance(funcOrCoord, FunctionWithBasis):
        assert(funcOrCoord.isTest() or funcOrCoord.isUnknown())
        isFunc = True
        id = funcOrCoord.funcID()
      else:
        isFunc = False
        id = funcOrCoord.direction()
        # We should never differentiate a coord expr at this point
        assert(isinstance(op, _IdentityOp))

      self.data = (isFunc, id, op)
      self._name = funcOrCoord.name()

    def __eq__(self, other):
        assert(isinstance(other, DerivSpecifier))
        return self.data == other.data;
       

    def __le__(self, other):
        assert(isinstance(other, DerivSpecifier))
        
        return self.data < other.data
    
    def isFunc(self):
        return self.data[0]
    
    def isCoord(self):
        return not self.isFunc()

    def id(self):
        return self.data[1]

    def op(self):
        return self.data[2]
    
    def name(self):
        return _name

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        return 'DerivSpecifier(isFunc={}, id={}, op={}, name={})'.format(
            self.data[0], self.data[1], self.data[2], self._name)

    def __str__(self):
        if self.op()==_IdentityOp():
            return self._name
        else:
            return '{}({})'.format(self.op(), self._name)


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

    print(s)

    mu = MultipleDeriv([I_u, dx_v, I_u])
    nu = MultipleDeriv([grad_v, I_v, x])
    print('mu=',mu)
    print('nu=',nu)

    m = {
        mu : 'A',
        nu : 'B'
    }

    print('m=',m)
