from abc import ABC, abstractmethod

import numpy as np


class AbstractProxCalculator(ABC):

    @abstractmethod
    def calculate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def is_eligible(self, type: str) -> bool:
        raise NotImplementedError("Abstract method shall not be invoked!")
