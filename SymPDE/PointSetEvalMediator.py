import numpy as np

class PointSetEvalMediator(EvalMediatorBase):

    def __init__(self):
        pass
        self._coords = None
        self._points = None
        self._funcValCache = {}
        self._basisValCache = {}

    def setPoints(self, points):
        '''
        Set the coordinate points for this evaluator. They should be in a
        numpy array with x coordinates in column 0, y coordinates in column 1,
        and so on.
        '''

        assert(isinstance(points, np.ndarray))
        shape = points.shape
        nPts = shape[0]
        dim = shape[1]

        self._points = points
        self._coords = []

        for i in range(dim):
            self._coords.append(points[:,i])


    def evalCoord(self, coordDir):
        return self._points[coordDir]

    def evalDiscreteFunctionElement(self, funcElem, diffOp):

        raise NotImplementedError('evalDiscreteFunctionElement')


    def evalDiscreteFunction(self, func, diffOp):

        assert(isinstance(func, ScalarDiscreteFunction, VectorDiscreteFunction))

        if func not in self._funcValCache:

            basis = func.basis()
            coeffs = func.coeffs()

            if (basis, diffOp) not in self._basisValCache:
                basisVals = basis.eval(pts, diffOp)
                self._basisValCache[(basis, diffOp)] = basis.eval(pts, diffOp)
            else:
                basisVals = self._basisValCache[(basis, diffOp)]


            funcVals = np.dot(basisVals, coeffs)
            self.funcValsCache_[func] = funcVals
        else
            funcVals = self.funcValsCache_[func]

        return funcVals
