from prox.abstract_prox_calculator import AbstractProxCalculator

from typing import Optional


class ProxCalculatorProvider:

    def __init__(self) -> None:
        self._calculators = []
        children = AbstractProxCalculator.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._calculators.append(child())

    """ Public methods """

    def provide_calculator(self, type: str) -> Optional[AbstractProxCalculator]:

        if len(self._calculators) == 0:
            return None

        for calculator in self._calculators:
            if calculator.is_eligible(type):
                return calculator

        return None
