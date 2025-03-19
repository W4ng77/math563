from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm

class DeblurAlgorithmChambollePock(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        pass

    def should_use(self, algorithm: str) -> bool:
        return algorithm == "chambollepock"