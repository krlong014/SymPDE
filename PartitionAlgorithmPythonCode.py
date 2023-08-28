from collections import Iterable
import itertools

def flattenList(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flattenList(item):
                 yield x
         else:        
             yield item

def ListWithoutRepetitions(l):
    new_k = []
    for elem in l:
        if elem not in new_k:
            new_k.append(elem)
            
    return new_k

def partitionreal(N):
    if N == 1:
        return [1]
    
    partArray = []
    for i in range(N):
        for j in range(len(partitionreal(i))):
           partArray.append([N - i] + [partitionreal(i)[j]])
    
    return partArray

def partitionWithPermutations(N):
    tempArray = []
    partArray = partitionreal(N)

    for i in range(len(partArray)):
        s = [list(flattenList(partArray[i]))]
        if s not in tempArray:
            tempArray += s

    tempArray.sort(reverse=True)
    return [[N]] + tempArray

def partition(N):
    tempArray = []
    partArray = partitionreal(N)

    for i in range(len(partArray)):
        tempArray += [list(flattenList(partArray[i]))]
        tempArray[i].sort(reverse=True)

    tempArray.sort(reverse=True)
    tempArray = ListWithoutRepetitions(tempArray)
    return [[N]] + tempArray

val = 7

print(partition(val))
print(partitionWithPermutations(val))