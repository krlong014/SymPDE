import sys
from OrderedTuple import OrderedTuple


# --------------------------------------------------------------------
#
# The function nonEmptyArrangements finds the arrangments,
# neglecting permutations, of M items into n non-empty
# boxes for 1 <= n <= M. The result is returned as a dictionary with
# keys = n and values = list of arrangments with that n.
#
# Examples:
#
# arrangements of  ['a', 'b']
#	n= 1
#		i=   0: (('a', 'b'),)
#	n= 2
#		i=   0: (('a',), ('b',))
#
# arrangements of  ['a', 'b', 'c']
# n= 1
# 	i=   0: (('a', 'b', 'c'),)
# n= 2
# 	i=   0: (('a',), ('b', 'c'))
# 	i=   1: (('a', 'c'), ('b',))
# 	i=   2: (('a', 'b'), ('c',))
# n= 3
# 	i=   0: (('a',), ('b',), ('c',))
# ---------------------------------------------------------------
def nonEmptyArrangements(items):
    M = len(items)

    arr = listArrangements(M, items)

    tups = {}
    for n in range(0,M):
        tups[n+1]=set()

    for a in arr:
        nonEmpty = []
        for x in a:
            if len(x)>0:
                nonEmpty.append(tuple(x))

        count = len(nonEmpty)
        if count==0:
            continue
        nonEmpty.sort()
        #print('count={}, non-empty={}'.format(count,nonEmpty))
        ne = tuple(nonEmpty)

        tups[count].add(ne)


    return tups

# ---------------------------------------------------------------
# Code to place M items in N boxes.
# ---------------------------------------------------------------

def integerDigits(x, N, M):
    '''
    Return as a tuple the first M digits of x in base N integer arithmetic.
    It is assumed that 0 <= x <= N**M-1.
    '''


    assert(isinstance(M,int) and M>0)
    assert(isinstance(N,int) and N>0)
    assert(isinstance(x,int) and x>=0 and x<N**M)

    digits = []
    remainder = x
    for i in range(0,M):
        d = remainder % N
        digits.append(d)
        remainder = remainder // N

    return tuple(digits)


def listArrangements(N, items):

    M = len(items)
    arr = []

    for x in range(0,N**M):
        dig = integerDigits(x, N, M)
        curA = [[] for i in range(N)]

        #print('A=',curA)
        for valIndex,boxIndex in enumerate(dig):
            #print('box={}, item={}'.format(boxIndex, valIndex))
            curA[boxIndex].append(items[valIndex])
        arr.append(curA)

    return arr

def getFactorSets(items):
    maxOrder = 3
    order = len(items)
    assert( order > 0)
    assert( order <= maxOrder)

    rtn = None
    if order == 1:
        F1_1 = { OrderedTuple(items[0]) }
        rtn = {1: F1_1 }
    elif order == 2:
        F2_1 = { OrderedTuple([items[0], items[1]]) }
        F2_2 = { OrderedTuple(items[0]), OrderedTuple(items[1]) }
        rtn = {1: F2_1, 2: F2_2}
    elif order == 3:
        F3_1 = { OrderedTuple([items[0], items[1], items[2]]) },
        F3_2 = {
                (OrderedTuple([items[0], items[1]]), OrderedTuple(items[2])),
                (OrderedTuple([items[0], items[2]]), OrderedTuple(items[1])),
                (OrderedTuple([items[1], items[2]]), OrderedTuple(items[0]))
            }
        F3_3 = {OrderedTuple(items[0]), OrderedTuple(items[1]),
                    OrderedTuple(items[2])}
        rtn = {1: F3_1, 2: F3_2, 3: F3_3}

    return rtn



# ---------------------------------------------------------------
# Code to enumerate all N-tuples of indices 1 through P
# ---------------------------------------------------------------

def indexTuples(N,P):
    rtn = []

    if N==1:
        for i in range(0,P):
            rtn.append((i,))
        return rtn

    elif N==2:
        for i in range(0,P):
            for j in range(0,P):
                rtn.append((i,j))
        return rtn

    elif N==3:
        for i in range(0,P):
            for j in range(0,P):
                for k in range(0,P):
                    rtn.append((i,j,k))
        return rtn

    elif N==4:
        for i in range(0,P):
            for j in range(0,P):
                for k in range(0,P):
                    for ell in range(0,P):
                        rtn.append((i,j,k,ell))
        return rtn

    else:
        assert( N!=1 and N!=2 and N!=3 and N!=4 )


def writePartial(func, vars):
    d = len(vars)
    denom = ''
    for i in range(d):
        if i > 0:
            denom += '\\,'
        denom += '\\partial ' + vars[i]
    if d==1:
        return '\\frac{\\partial %s}{%s}' % (func,denom)
    return '\\frac{\\partial^%d %s}{%s}' % (d, func,denom)

def writeChainRule(nArgs,vars, of=sys.stdout):
    M = len(vars)
    P = nArgs
    args = ['a_%d' % i for i in range(1,P+1)]

    # Get all index tuples
    #print('------- finding index tuples')
    Q = {}
    for n in range(1,M+1):
        #print('finding Q for n=', n)
        Q[n] = indexTuples(n,P)
        #print('Q[{}]={}'.format(n,Q))


    # Get all derivative combinations
    #print('------- finding deriv combinations')
    H = nonEmptyArrangements(vars)



    print('\\begin{multline*}')

    print( writePartial('g', vars), '=')

    #print('length of Q = ', len(Q))
    count = 0
    for n in range(1,M+1):
        if n > 1:
            print('+')
        Q_n = Q[n]
        H_n = H[n]

        #print('Q[{}]={}'.format(n,Q_n))
        #print('H[{}]={}'.format(n,H_n))

        #print('\\left(')
        first = True
        for Q_n_ell in Q_n:
            qVars = [args[i] for i in Q_n_ell]
            if not first:
                print('+')
            first = False
            #print('qVars=', qVars)
            dg_da = writePartial('g', qVars)
            print(dg_da)
            print('\\left(')
            for k,H_n_k in enumerate(H_n):
                if k>0:
                    print('+')

                for j, u in enumerate(H_n_k):

                    da_dVars = writePartial(qVars[j], u)
                    print(da_dVars)
                    count += 1


            print('\\right)')
            if count > 5:
                count=0
                print('\\\\')


    print('\\end{multline*}')










def startLatex(title):
    start = '''
    \\documentclass[letter]{article}
    \\special{papersize=8.5in,11in}
    \\usepackage[margin=0.5in]{geometry}
    \\usepackage{amsmath}

    \\title{%s}
    \\begin{document}
    ''' % title

    return start

def endLatex():
    return '\\end{document}'





if __name__=='__main__':
    # Test case: place 4 items in 3 boxes

    # a1 = ['a',]
    # a2 = ['a','b']
    # a3 = ['a','b','c']
    # a4 = ['a','b','c','d']
    #
    #
    # for A in (a1,a2,a3,a4):
    #     print('{}'.format('='*60))
    #     print('arrangements of ', A)
    #
    #     arr = nonEmptyArrangements(A)
    #
    #     for n in arr:
    #         print('\tn=',n)
    #         tups = arr[n]
    #         for i,t in enumerate(tups):
    #             print('\t\ti={:4d}: {}'.format(i,t))
    #
    #
    # print('{}'.format('='*60))
    # print('Index arrangements')
    #
    # P=3
    # for N in range(1,4):
    #     arr = indexTuples(N,P)
    #     print('N={:3d}'.format(N))
    #     for i,a in enumerate(arr):
    #         print('\t{:4d}: {}'.format(i,a))
    #
    #
    # print(writePartial('f', ('x','y','z')))


    for p in (['u',], ['u','v'], ['u','v','w']):
        print('Factor sets for p=', p)
        F = getFactorSets(p)
        for n in range(1,len(p)+1):
            print('\tn={}:'.format(n))
            print('\t\tp[{:d}]={}'.format(n,F[n]))

    # print(startLatex('Chain rule test'))
    #
    # print('\\section{One variable, two args}')
    # writeChainRule(2, ['x'])
    #
    # print('\\section{Two variables, two args}')
    # writeChainRule(2, ['x','y'])
    #
    # print('\\section{Three variables, two args}')
    # writeChainRule(2, ['x','y', 'z'])
    #
    # print('\\section{Two variables, three args}')
    # writeChainRule(3, ['x','y'])
    #
    # print('\\section{Three variables, three args}')
    # writeChainRule(3, ['x','y', 'z'])
    #
    # print(endLatex())
