from DiffOp import (_Partial, _IdentityOp)
from BasisBase import ScalarBasisBase
import numpy as np

class FourierSineBasis(ScalarBasisBase):
    def __init__(self, n):
        super().__init__(n)

    def eval(self, pts, diffOp):
        assert(isinstance(diffOp, (_IdentityOp, _Partial)))


        rtn = np.zeros((len(pts), self.order()))

        for j in range(self.order()):
            if isinstance(diffOp, _IdentityOp):
                rtn[:,j] = np.sin((j+1)*pts)
            else:
                rtn[:,j] = (j+1)*np.cos((j+1)*pts)

        return rtn



if __name__=='__main__':


    import matplotlib.pyplot as plt

    n = 16
    bas = FourierSineBasis(n)
    c = np.zeros(n)

    for i in range(n):
        if i%2==0:
            c[i]=4/np.pi/(i+1)

    x = np.linspace(-np.pi, np.pi, 360)
    phi = bas.eval(x, _IdentityOp())
    phi1 = bas.eval(x, _Partial(0))
    f = np.dot(phi, c)
    f1 = np.dot(phi1, c)

    plt.plot(x, f, 'k-', x, f1, 'r-')
    plt.show()
