from . ExprWithChildren import UnaryExpr

class DiffOp(UnaryExpr):

    def __init__(self, op, arg):
        
        from . HungryDiffOp import HungryDiffOp

        assert(isinstance(op, HungryDiffOp))
        if op.acceptShape(arg.shape()):
            myShape = op.outputShape(arg.shape())
        else:
            raise ValueError('Undefined DiffOp action: {} acting on {}'.format(op, arg))

        super().__init__(arg, myShape)
        self._op = op

    def op(self):
        return self._op


    def __str__(self):
        return '{}({})'.format(self.op().__str__(), self.arg().__str__())



class DiffOpOnFunction(DiffOp):
    '''
    This class is a specialization of diff op to the case where the argument is a 
    function.
    '''

    def __init__(self, op, arg):
      '''Constructor. All that's done is to check type and then 
        construct the superclass'''    
      from . FunctionWithBasis import FunctionWithBasis

      assert(isinstance(arg, FunctionWithBasis))
      super().__init__(op, arg)

    def funcID(self):
        return self.arg().funcID()

    def isTest(self):
        return self.arg().isTest()

    def isUnknown(self):
        return self.arg().isUnknown()

    def isDiscrete(self):
        return self.arg().isDiscrete()


    def isIndependentOf(self, u):
        if self.arg()==u:
            return False
        return True

    def isLinearInTests(self):
        return self.isTest()

