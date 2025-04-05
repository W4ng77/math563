from prox.abstract_prox_calculator import AbstractProxCalculator
from utils.constants import *
import numpy as np

class ProxCalculatorL2(AbstractProxCalculator):
    def __init__(self):
        self.t = None  # Step size
        self.b = None  # Blurred image

    def calculate(self, x: np.ndarray) -> np.ndarray:
        """
        Proximal operator for ||x - b||_2^2.
        Returns: (2t * b + x) / (2t + 1)
        """
        if self.t is None or self.b is None:
            raise ValueError("ProxCalculatorL2: 't' and 'b' must be set before calling calculate.")

        numerator = 2 * self.t * self.b + x
        denominator = 2 * self.t + 1
        return numerator / denominator

    def is_eligible(self, type: str) -> bool:
        return type == L2_PROX
