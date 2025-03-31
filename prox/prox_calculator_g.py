from prox.abstract_prox_calculator import AbstractProxCalculator
from prox.prox_calculator_provider import ProxCalculatorProvider
from utils.constants import *
import numpy as np

class ProxCalculatorG(AbstractProxCalculator):
    def __init__(self, norm_type: str, b: np.ndarray, t: float, gamma: float):
        """
        Initialize ProxCalculatorG with required parameters.

        Parameters:
        - norm_type: 'l1' or 'l2' (used to select the fidelity term)
        - b: blurred image
        - t: step size
        - gamma: denoising parameter for iso prox
        """
        self.norm_type = norm_type
        self.b = b
        self.t = t
        self.gamma = gamma

        # Load calculators from provider
        self.provider = ProxCalculatorProvider()
        self.fidelity_calculator = self.provider.provide_calculator(
            L1_PROX if norm_type == "l1" else L2_PROX
        )
        self.iso_calculator = self.provider.provide_calculator(ISO_PROX)

        # Set parameters directly (these fields should exist in each class)
        self.fidelity_calculator.t = t
        self.fidelity_calculator.b = b
        self.iso_calculator.t = t
        self.iso_calculator.gamma = gamma

    def calculate(self, x: np.ndarray) -> np.ndarray:
        """
        Apply prox_g to x ∈ ℝ^{H×W×3}, where:
            x[:,:,0] = y1 (l1/l2 fidelity)
            x[:,:,1] = y2
            x[:,:,2] = y3
        """
        y1 = x[:, :, 0]
        y2 = x[:, :, 1]
        y3 = x[:, :, 2]

        # Apply prox for l1/l2 fidelity
        y1_new = self.fidelity_calculator.calculate(y1)

        # Apply iso prox to y2 and y3
        y_iso_input = np.stack([y2, y3], axis=-1)
        y_iso_output = self.iso_calculator.calculate(y_iso_input)
        y2_new = y_iso_output[:, :, 0]
        y3_new = y_iso_output[:, :, 1]

        # Combine result
        y_new = np.stack([y1_new, y2_new, y3_new], axis=-1)
        return y_new

    def is_eligible(self, type: str) -> bool:
        return type == G_PROX
