from functools import total_ordering

@total_ordering
class OrderedTuple:
    """
    Class OrderedTuple bundles orderable objects {e_i} into
    a tuple ordered so that e_1 <= e_2 <= ... <= e_N.

    Example: OrderedTuple([3,4,1]) stores (1,3,4) in self.elems.

    OrderedTuple objects have a total ordering defined lexicographically, and
    are hashable by the internal tuple.
    """
    def __init__(self, elems):
        if len(elems)==1:
            self.elems=(elems,)
        else:
            tmp = []
            for e in elems:
                tmp.append(e)
            self.elems=tuple(sorted(tmp))

    def __getitem__(self, i):
        return self.elems[i]

    def __len__(self):
        return len(self.elems)

    def __str__(self):
        rtn = '('
        for i,e in enumerate(self.elems):
            if i>0:
                rtn += ', '
            rtn += '{}'.format(e)
        rtn += ')'
        return rtn

    def __repr__(self):
        return 'OrderedTuple(' + str(self) + ')'



    def __hash__(self):
        return hash(self.elems)

    def __eq__(self, other):
        for me, you in zip(self.elems, other.elems):
            if me!=you:
                return False
        return True

    def __le__(self, other):
        for me, you in zip(self.elems, other.elems):
            if me < you:
                return True
            if me > you:
                return False
        return False


if __name__=='__main__':

    A = OrderedTuple([2,1])
    B = OrderedTuple([3,4])

    print('A=', A)
    print('B=', B)

    s = {A,B}
    print(s)

    x = OrderedTuple([B,A])
    print(x)

    d = {'Bob':1, 'Joe':2}
    print(d)
    s2 = {1,2,3,4}
    print(s2)
