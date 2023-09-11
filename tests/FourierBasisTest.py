import pytest
import numpy as np
from numpy.linalg import norm

from SymPDE.DiffOp import _IdentityOp, _Partial
from SymPDE.Spectral.FourierBasis import FourierSineBasis

class TestFourier:

    def test_evalFuncAndDeriv(self):

        N = 3
        bas = FourierSineBasis(N)

        X = np.array([0.25,0.5])

        vals = bas.eval(X, _IdentityOp())
        print('\nvals=', vals)


        exactVals = np.zeros([len(X), N])
        for i in range(len(X)):
            for n in range(N):
                exactVals[i,n] = np.sin((n+1)*X[i])
        print('\nexact vals=', exactVals)

        dvals = bas.eval(X, _Partial(0))
        print('\ndvals=', dvals)

        exactdVals = np.zeros([len(X), N])
        for i in range(len(X)):
            for n in range(N):
                exactdVals[i,n] = (n+1)*np.cos((n+1)*X[i])
        print('\nexact derivs=', exactdVals)


        print('\nvals shape={}, exactVals shape={}'
            .format(vals.shape, exactVals.shape))

        err = vals-exactVals
        derr = dvals-exactdVals
        print('\nval error = ', err)
        print('\ndval error = ', derr)

        errNorm = norm(err, ord='fro')
        derrNorm = norm(derr, ord='fro')

        print('\nval error norm = ', errNorm)
        print('\ndval error norm = ', derrNorm)

        tol = 1.0e-14
        assert(errNorm <= tol and derrNorm <= tol)
