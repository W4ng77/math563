from prox.abstract_prox_calculator import AbstractProxCalculator
from utils.constants import *
import numpy as np

class ProxCalculatorL1(AbstractProxCalculator):
    def __init__(self, t: float, b: np.ndarray):
        self.t = t
        self.b = b

    def calculate(self, x: np.ndarray) -> np.ndarray:
        z = x - self.b
        shrink = np.sign(z) * np.maximum(np.abs(z) - self.t, 0)
        return shrink + self.b

    def is_eligible(self, type: str) -> bool:
        return type == L1_PROX
