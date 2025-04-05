from prox.abstract_prox_calculator import AbstractProxCalculator
from prox.prox_calculator_l1 import ProxCalculatorL1
from prox.prox_calculator_l2 import ProxCalculatorL2
from prox.prox_calculator_iso import ProxCalculatorIso
from utils.constants import *
import numpy as np


class ProxCalculatorG(AbstractProxCalculator):
    def __init__(self, norm_type: str, b: np.ndarray, t: float, gamma: float):
        self.norm_type = norm_type
        self.b = b
        self.t = t
        self.gamma = gamma

    def calculate(self, x: np.ndarray) -> np.ndarray:
        y1 = x[:, :, 0]
        y2 = x[:, :, 1]
        y3 = x[:, :, 2]

        # Fidelity term prox
        if self.norm_type == "l1":
            fidelity_calculator = ProxCalculatorL1(self.t, self.b)
        else:
            fidelity_calculator = ProxCalculatorL2(self.t, self.b)

        y1_new = fidelity_calculator.calculate(y1)

        # Iso prox for y2 and y3
        iso_input = np.stack([y2, y3], axis=-1)
        iso_calculator = ProxCalculatorIso(self.t, self.gamma)
        iso_output = iso_calculator.calculate(iso_input)
        y2_new = iso_output[:, :, 0]
        y3_new = iso_output[:, :, 1]

        return np.stack([y1_new, y2_new, y3_new], axis=-1)

    def is_eligible(self, type: str) -> bool:
        return type == G_PROX
