

class EvalContext:

    def __init__(self, regionAndQuad, requiredDiffOrders,
                 verb=0):
        self.verb = 0
        self.regionAndQuad = regionAndQuad
        self.requiredDiffOrders = frozenset(requiredDiffOrders)

    def verb(self):
        return self.verb

    def __str__(self):
        return 'EvalContext(RQ={}, orders={})'.format(
            self.regionAndQuad, self.requiredDiffOrders
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, EvalContext):
            return False

        return (self.regionAndQuad==other.regionAndQuad and
            self.requiredDiffOrders == other.requiredDiffOrders)

    def __hash__(self):
        return hash((self.regionAndQuad, self.requiredDiffOrders))


if __name__=='__main__':

    c1 = EvalContext(1, {1,2})
    c2 = EvalContext(1, {1})

    s = {c1, c2}

    print(s)
