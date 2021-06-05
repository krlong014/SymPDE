

class DiscreteSpaceBase:

    def __init__(self, basis):
        self._basis = basis

    def basis(self):
        return self._basis
