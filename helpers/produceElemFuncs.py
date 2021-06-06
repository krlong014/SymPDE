

template = """
class %sFunc(UnivariateFuncExpr):
    def __init__(self, x):
        super().__init__('%s', x)

    def deriv(self, x):
        return %s


def %s(x):
    return %sFunc(x)
"""


funcs = (
    ['Exp', 'Exp(x)'],
    ['Log', '1/x'],
    ['Sqrt', '1/2/Sqrt(x)'],
    ['Cos', '-Sin(x)'],
    ['Sin', 'Cos(x)'],
    ['Tan', '1/Cos(x)**2'],
    ['Cosh', 'Sinh(x)'],
    ['Sinh', 'Cosh(x)'],
    ['Tanh', '1/Cosh(x)**2'],
    ['ArcCos', '-1/Sqrt(1-x**2)'],
    ['ArcSin', '1/Sqrt(1-x**2)'],
    ['ArcTan', '1/(1+x**2)'],
    ['ArcCosh', '1/Sqrt(x**2-1)'],
    ['ArcSinh', '1/Sqrt(1+x**2)'],
    ['ArcTanh', '1/(1-x**2)']
)

for pair in funcs:
    f = pair[0]
    df = pair[1]
    print(template % (f, f, df, f, f))
