from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm
from algorithm.deblur_algorithm_admm import DeblurAlgorithmAdmm
from algorithm.deblur_algorithm_chambolle_pock import DeblurAlgorithmChambollePock
from algorithm.deblur_algorithm_douglas_rachford_primal import DeblurAlgorithmDouglasRachfordPrimal
from algorithm.deblur_algorithm_douglas_rachford_primal_dual import DeblurAlgorithmDouglasRachfordPrimalDual

from typing import Optional


class DeblurAlgorithmFactory:
    """ Initialize """

    def __init__(self) -> None:
        self._algorithms = []
        children = AbstractDeblurAlgorithm.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._algorithms.append(child())

    """ Public methods """

    def get_algorithm(self, algo: str) -> Optional[AbstractDeblurAlgorithm]:

        if len(self._algorithms) == 0:
            return None

        for algorithm in self._algorithms:
            if algorithm.should_use(algo):
                return algorithm

        return None
