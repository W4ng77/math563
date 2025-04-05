from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm
from algorithm.deblur_algorithm_admm import DeblurAlgorithmAdmm
from algorithm.deblur_algorithm_chambolle_pock import DeblurAlgorithmChambollePock
from algorithm.deblur_algorithm_douglas_rachford_primal import DeblurAlgorithmDouglasRachfordPrimal
from algorithm.deblur_algorithm_douglas_rachford_primal_dual import DeblurAlgorithmDouglasRachfordPrimalDual

from typing import Optional


class DeblurAlgorithmFactory:
    """Factory class to provide appropriate deblurring algorithm instance."""

    def __init__(self) -> None:
        self._algorithms = []
        children = AbstractDeblurAlgorithm.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._algorithms.append(child())

    def get_algorithm(self, algo: str) -> Optional[AbstractDeblurAlgorithm]:
        if len(self._algorithms) == 0:
            return None
        for algorithm in self._algorithms:
            if algorithm.should_use(algo):
                return algorithm
        return None


# === Module-level function for convenience === #
_factory = DeblurAlgorithmFactory()

def get_algorithm(algo: str) -> Optional[AbstractDeblurAlgorithm]:
    """
    Global function to fetch the desired deblurring algorithm.

    :param algo: The algorithm name string (e.g., 'admm', 'chambollepock', etc.)
    :return: Corresponding AbstractDeblurAlgorithm instance or None
    """
    return _factory.get_algorithm(algo)
