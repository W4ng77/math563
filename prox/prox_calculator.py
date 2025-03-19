from prox.prox_calculator_provider import ProxCalculatorProvider

import numpy as np


class ProxCalculator:

    def __init__(self):
        self._provider = ProxCalculatorProvider()

    def calculate(self, x: np.ndarray, type: str) -> np.ndarray:
        calculator = self._provider.provide_calculator(type)
        if calculator is None:
            raise ValueError(f"Calculator for type {type} not found")
        return calculator.calculate(x)
