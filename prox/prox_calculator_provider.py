from prox.abstract_prox_calculator import AbstractProxCalculator
from prox.prox_calculator_box import ProxCalculatorBox
from prox.prox_calculator_l1 import ProxCalculatorL1
from prox.prox_calculator_l2 import ProxCalculatorL2
from prox.prox_calculator_iso import ProxCalculatorIso
from prox.prox_calculator_g import ProxCalculatorG

from utils.constants import *
from typing import Optional


class ProxCalculatorProvider:
    """Factory for creating ProxCalculator objects with required parameters."""

    def provide_calculator(self, type: str, **kwargs) -> Optional[AbstractProxCalculator]:
        if type == BOX_PROX:
            return ProxCalculatorBox()
        elif type == L1_PROX:
            return ProxCalculatorL1(kwargs["t"], kwargs["b"])
        elif type == L2_PROX:
            return ProxCalculatorL2(kwargs["t"], kwargs["b"])
        elif type == ISO_PROX:
            return ProxCalculatorIso(kwargs["t"], kwargs["gamma"])
        elif type == G_PROX:
            return ProxCalculatorG(
                norm_type=kwargs["norm_type"],
                b=kwargs["b"],
                t=kwargs["t"],
                gamma=kwargs["gamma"]
            )
        for calculator in self._calculators:
            if calculator.is_eligible(type):
                return calculator
        else:
            return None

