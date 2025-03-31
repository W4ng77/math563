from prox.abstract_prox_calculator import AbstractProxCalculator
from utils.constants import *
import numpy as np

class ProxCalculatorBox(AbstractProxCalculator):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        """
        Projects each element of x onto the interval [0, 1].
        Equivalent to prox of the indicator function of the box constraint [0, 1].
        """
        return np.clip(x, 0, 1)

    def is_eligible(self, type: str) -> bool:
        return type == BOX_PROX
