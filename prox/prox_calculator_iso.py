from prox.abstract_prox_calculator import AbstractProxCalculator

from utils.constants import *
import numpy as np


class ProxCalculatorIso(AbstractProxCalculator):
    def calculate(self, x: np.ndarray) -> np.ndarray:
        pass

    def is_eligible(self, type: str) -> bool:
        return type == ISO_PROX
