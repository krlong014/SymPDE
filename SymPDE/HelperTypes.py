
# ---------------------------------------------------------------------
# ComputationType indicates what quantities are requested from
# a symbolic expression
# ---------------------------------------------------------------------

class ComputationType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'ComputationType({})'.format(self.name)

    def __eq__(self, other):
        return self.name==other.name

    def __hash__(self):
        return hash(self.name)

class MatrixAndVector(ComputationType):
    def __init__(self):
        super().__init__('MatrixAndVector')

class VectorOnly(ComputationType):
    def __init__(self):
        super().__init__('VectorOnly')

class FunctionalOnly(ComputationType):
    def __init__(self):
        super().__init__('FunctionalOnly')

class FunctionalAndGradient(ComputationType):
    def __init__(self):
        super().__init__('FunctionalAndGradient')

class Sensitivities(ComputationType):
    def __init__(self):
        super().__init__('Sensitivities')



# ---------------------------------------------------------------------
# DerivState describes whether a functional derivative is structurally
# zero, spatially constant, or spatially variable
# ---------------------------------------------------------------------

class DerivState:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'DerivState({})'.format(self.name)

    def __eq__(self, other):
        return self.name==other.name

    def __hash__(self):
        return hash(self.name)

class StructuralZero(DerivState):
    def __init__(self):
        super().__init__('StructuralZero')

class SpatialConstant(DerivState):
    def __init__(self):
        super().__init__('SpatialConstant')

class SpatialVariable(DerivState):
    def __init__(self):
        super().__init__('SpatialVariable')


# ---------------------------------------------------------------------
# DerivSubsetType indicates whether a set indicates all nonzero
# functional derivatives, spatially constant nonzero FDs, spatially
# variable nonzero FDs, or required nonzero FDs
# ---------------------------------------------------------------------

class DerivSubsetType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'DerivSubsetType({})'.format(self.name)

    def __eq__(self, other):
        return self.name==other.name

    def __hash__(self):
        return hash(self.name)

class AllNonzeros(DerivSubsetType):
    def __init__(self):
        super().__init__('AllNonzeros')

class ConstantNonzeros(DerivSubsetType):
    def __init__(self):
        super().__init__('ConstantNonzeros')

class VariableNonzeros(DerivSubsetType):
    def __init__(self):
        super().__init__('VariableNonzeros')

class RequiredNonzeros(DerivSubsetType):
    def __init__(self):
        super().__init__('RequiredNonzeros')




if __name__=='__main__':
    t1 = MatrixAndVector()
    t2 = FunctionalOnly()

    s = {t1, t2}

    print(s)

    s1 = StructuralZero()
    s2 = SpatialConstant()
    s = {s1, s2}
    print(s)
