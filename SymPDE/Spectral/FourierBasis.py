from .. BasisBase import ScalarBasisBase
from .. HungryDiffOp import _Partial, _IdentityOp
import numpy as np

class FourierSineBasis(ScalarBasisBase):
    '''Fourier sine basis on [0,pi]'''

    def __init__(self, order):
        super().__init__(order)

    def eval(self, pts, diffOp):
        c1 = np.cos(pts)
        s1 = np.sin(pts)
        N = self.order()

        c = np.ones([len(pts), N])
        s = np.ones([len(pts), N])

        for i in range(len(pts)):
            c[i,0]=c1[i]
            s[i,0]=s1[i]
            for n in range(1,N):
                c[i,n]=c1[i]*c[i,n-1] - s1[i]*s[i,n-1]
                s[i,n]=s1[i]*c[i,n-1] + c1[i]*s[i,n-1]

        if isinstance(diffOp, _IdentityOp):
            return s

        if not isinstance(diffOp, _Partial):
            raise ValueError('Expected diffOp=_Partial, got {}'.format(diffOp))

        if diffOp.direction() != 0:
            raise ValueError('Expected diffOp direction=0, got {}'
                .format(diffOp.direction()))

        rtn = np.zeros((len(pts), N))

        for i in range(len(pts)):
            for n in range(N):
                rtn[i,n] = (n+1)*c[i,n]

        return rtn
