from prox.abstract_prox_calculator import AbstractProxCalculator
from utils.constants import *
import numpy as np

class ProxCalculatorIso(AbstractProxCalculator):
    def __init__(self, t: float, gamma: float):
        self.t = t
        self.gamma = gamma

    def calculate(self, x: np.ndarray) -> np.ndarray:
        y2 = x[:, :, 0]
        y3 = x[:, :, 1]

        if self.gamma == 0:
            prox2 = y2
            prox3 = y3
        else:
            magnitude = np.sqrt(y2**2 + y3**2)
            factor = 1 - self.t * self.gamma / np.maximum(magnitude, self.t * self.gamma)
            prox2 = y2 * factor
            prox3 = y3 * factor

        return np.stack([prox2, prox3], axis=-1)

    def is_eligible(self, type: str) -> bool:
        return type == ISO_PROX

