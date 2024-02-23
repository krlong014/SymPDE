from SymPDE.Coordinate import Coordinate

x = Coordinate(0)
y = Coordinate(1)
f = x
g = -y
h = f + g

context = 'bob'

eval = h.makeEval(context)

print('\n\nEvaluator for {} is {}\n\n'.format(h, eval))