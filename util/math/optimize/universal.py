
class OptimizationError(Exception):

    def __init__(self, result):
        self.result = result

    def __str__(self):
        return str(self.result)
