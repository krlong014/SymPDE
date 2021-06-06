

class DiscreteSpaceBase:

    def __init__(self, bases):
        self._bases = bases

    def basis(self, func):
        return self._bases[func]

    def numFuncs(self):
        return len(self._bases)
