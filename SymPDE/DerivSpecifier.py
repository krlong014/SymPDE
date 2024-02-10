from functools import total_ordering
from SymPDE.OrderedTuple import OrderedTuple

# KRL to do: distinguist test vs unknown vs parameter

@total_ordering
class DerivSpecifier:
    from SymPDE.HungryDiffOp import _IdentityOp

    def __init__(self, funcOrCoord, op=_IdentityOp()):


      from SymPDE.Coordinate import Coordinate
      from SymPDE.FunctionWithBasis import FunctionWithBasis
      from SymPDE.HungryDiffOp import HungryDiffOp, _IdentityOp

      assert(isinstance(op, HungryDiffOp))

      assert(isinstance(funcOrCoord, FunctionWithBasis) 
              or isinstance(funcOrCoord, Coordinate))

      isTest = False
      if isinstance(funcOrCoord, FunctionWithBasis):
        assert(funcOrCoord.isTest() or funcOrCoord.isUnknown())
        isFunc = True
        if funcOrCoord.isTest():
            isTest = True
        id = funcOrCoord.funcID()
      else:
        isFunc = False
        id = funcOrCoord.direction()
        # We should never differentiate a coord expr at this point
        assert(isinstance(op, _IdentityOp))

      self._data = (isFunc, id, isTest, op)
      self._name = funcOrCoord.name()

    def __eq__(self, other):
        assert(isinstance(other, DerivSpecifier))
        return self._data == other._data;
       

    def __le__(self, other):
        assert(isinstance(other, DerivSpecifier))
        
        return self._data < other._data
    
    def isFunc(self):
        return self._data[0]
    
    def isCoord(self):
        return not self.isFunc()
    
    def isTestFunction(self):
        return self.isFunc() and self._data[3]
    
    def isUnknownFunction(self):
        return self.isFunc() and not self._data[3]

    def id(self):
        return self._data[1]

    def op(self):
        return self._data[2]
    
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self._data)

    def __repr__(self):
        return 'DerivSpecifier(isFunc={}, id={}, op={}, isTest={}, name={})'.format(
            self.isFunc(), self.id(), self.op(), self.isTest(), self.name())

    def __str__(self):
      from SymPDE.HungryDiffOp import _IdentityOp
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
    
    from SymPDE.FunctionWithBasis import UnknownFunction, TestFunction 
    from SymPDE.Coordinate import Coordinate
    from SymPDE.Lagrange import Lagrange
    from SymPDE.HungryDiffOp import _Partial, _Gradient

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
