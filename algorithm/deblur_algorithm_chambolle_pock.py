from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm
from prox.prox_calculator_provider import ProxCalculatorProvider
from utils.constants import *

import numpy as np

class DeblurAlgorithmChambollePock(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        # === Required inputs ===
        b = kwargs["b"]  # blurred image
        init_vectors = kwargs["init_vectors"]  # [x, y, z]
        problem = kwargs["problem"]  # "l1" or "l2"
        i = kwargs["params"]  # {"t": t, "s": s, "maxiter": maxiter, "gammal1": ..., ...}
        applyA = kwargs["applyA"]      # function: A(x)
        applyAT = kwargs["applyAT"]    # function: A^T(y)

        # === Initialization ===
        x, y, z = init_vectors
        t = i["tchambollepock"]
        s = i["schambollepock"]
        gamma = i["gammal1"] if problem == "l1" else i["gammal2"]

        # === Load Proximal Operators ===
        provider = ProxCalculatorProvider()
        prox_f = provider.provide_calculator(BOX_PROX)
        # prox_g = provider.provide_calculator(G_PROX)
        prox_g = provider.provide_calculator(
            G_PROX,
            norm_type=kwargs["norm_type"],
            b=kwargs["b"],
            t=kwargs["t"],
            gamma=kwargs["gamma"]
        )

        # Set params for prox_f (x step)
        prox_f.t = t

        # Set params for prox_g (used via Moreau identity)
        prox_g.t = 1 / s
        prox_g.b = b
        prox_g.gamma = gamma
        prox_g.norm_type = problem

        # === Main Iteration ===
        for _ in range(i["maxiter"]):
            # y^{k+1} = prox_{s g*}( y^k + s A z^k ) via Moreau identity
            v = y + s * applyA(z)
            y = v - s * prox_g.calculate(v / s)

            # x^{k+1} = prox_{t f}( x - t A^T y )
            x_prev = x
            x = prox_f.calculate(x - t * applyAT(y))

            # z^{k+1} = 2x^{k+1} - x^k
            z = 2 * x - x_prev

        return x

    def should_use(self, algorithm: str) -> bool:
        return algorithm == ALGORITHM_CHAMBOLLE_POCK
