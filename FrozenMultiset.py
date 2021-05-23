# FrozenMultiset is an immutable multiset suitable for use as keys in
# dictionaries or sets.

class FrozenMultiset:
    # Construct from a list or tuple of entries, possibly including
    # duplicates.
    def __init__(self, input):

        if isinstance(input, (tuple,list)):
            tmp = {}
            for x in sorted(input):
                if x in tmp.keys():
                    tmp[x]+=1
                else:
                    tmp[x]=1
        else:
            raise ValueError('FrozenMultiset ctor expected a list or tuple ',
            'got ', input)

        keyAgg = []
        data = []
        for k,v in tmp.items():
            keyAgg.append(k)
            data.append((k,v))
        self.keyAgg = tuple(keyAgg)
        self.data = tuple(data)

    # Find the multiplicity of an item in a FrozenMultiset. If it doesn't
    # contain the item, return 0.
    def count(self, x):
        for k,v in self.data:
            if k==x: return v
        return 0

    # Determine whether a FrozenMultiset contains a given value
    def contains(self, x):
        return x in self.keyAgg

    # Make a new FrozenMultiset with one more entry added.
    def copyAndAppend(self, y):
        # This isn't the most efficient implementation, but it's the
        # easiest to program.
        tmp = []
        for X in self.data:
            for i in range(X[1]):
                tmp.append(X[0])
        tmp.append(y)
        return FrozenMultiset(tmp)

    # Write to human-readable string
    def __str__(self):
        rtn = '{'
        for i,X in enumerate(self.data):
            if i != 0:
                rtn = rtn + ', '
            rtn = rtn + '(%s,%d)' % X
        rtn = rtn + '}'
        return rtn

    # Use lexicographic comparison.
    def __lt__(self, other):
        if not isinstance(other, FrozenMultiset):
            raise ValueError('comparison between multiset and non-multiset')

        for X1,X2 in zip(self.data, other.data):
            if X1[0]<X2[0]: return True
            if X1[0]>X2[0]: return False
            if X1[1]<X2[1]: return True
            if X1[1]>X2[1]: return False
        return False

    # Test equality with another FrozenMultiset
    def __eq__(self, other):
        if not isinstance(other, FrozenMultiset):
            raise ValueError('comparison between multiset and non-multiset')

        for X1,X2 in zip(self.data, other.data):
            if not X1[0] == X2[0]: return False
            if not X1[1] == X2[1]: return False

        return True

    # Hash a FrozenMultiset.
    def __hash__(self):
        return self.__str__().__hash__()



if __name__=='__main__':

    f1 = FrozenMultiset(['a','b','a'])
    f2 = FrozenMultiset(['a','b','c'])
    f3 = FrozenMultiset(['a','a','b'])

    print('f1=', f1, '\nf2=', f2, '\nf3=', f3)

    print('creating multiset of multisets\n\n\n')
    g = FrozenMultiset((f1, f2, f3))
    print(g)

    print('Count of \'a\' in f1 is ', f1.count('a'))
    print('Count of \'c\' in f1 is ', f1.count('c'))

    f4 = f3.copyAndAppend('d')
    f5 = f3.copyAndAppend('b')
    print('f4=', f4)
    print('f5=', f5)
