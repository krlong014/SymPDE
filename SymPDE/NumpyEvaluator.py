from EvaluationTree import EvaluatorNode

class NumpyEvaluator(EvaluatorNode):

    @singledispatchmethod
    def evaluate(self, arg, ):
        raise NotImplementedError('unknown type in makeNode')
