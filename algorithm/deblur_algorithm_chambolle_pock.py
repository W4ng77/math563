from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm

from utils.constants import *

class DeblurAlgorithmChambollePock(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        pass

    def should_use(self, algorithm: str) -> bool:
        return algorithm == ALGORITHM_CHAMBOLLE_POCK