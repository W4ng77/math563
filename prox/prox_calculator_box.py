from prox.abstract_prox_calculator import AbstractProxCalculator

from utils.constants import *
import numpy as np


class ProxCalculatorBox(AbstractProxCalculator):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        pass

    def is_eligible(self, type: str) -> bool:
        return type == BOX_PROX
