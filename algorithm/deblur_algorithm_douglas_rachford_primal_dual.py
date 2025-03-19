from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm

from utils.constants import *


class DeblurAlgorithmDouglasRachfordPrimalDual(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        pass

    def should_use(self, algorithm: str) -> bool:
        return algorithm == ALGORITHM_DOUGLAS_RACHFORD_PRIMAL_DUAL
