from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm
from prox.prox_calculator_provider import ProxCalculatorProvider
from utils.constants import *

import numpy as np


class DeblurAlgorithmAdmm(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        b = kwargs["b"]
        init_vectors = kwargs["init_vectors"]  # [u, y, w, z]
        problem = kwargs["problem"]
        i = kwargs["params"]
        applyA = kwargs["applyA"]
        applyAT = kwargs["applyAT"]
        invertMatrixT = kwargs["invertMatrixT"]

        u, y, w, z = init_vectors
        t = i["tadmm"]
        rho = i["rhoadmm"]
        gamma = i["gammal1"] if problem == "l1" else i["gammal2"]

        provider = ProxCalculatorProvider()
        prox_f = provider.provide_calculator(BOX_PROX)
        prox_g = provider.provide_calculator(
            G_PROX, norm_type=problem, b=b, t=1 / t, gamma=gamma
        )

        for _ in range(i["maxiter"]):
            x = invertMatrixT(u + applyAT(y) - (1 / t) * (w + applyAT(z)))
            u_new = prox_f.calculate(rho * x + (1 - rho) * u + w / t)
            y_new = prox_g.calculate(rho * applyA(x) + (1 - rho) * y + z / t)
            w = w + t * (x - u_new)
            z = z + t * (applyA(x) - y_new)
            u = u_new
            y = y_new

        xSol = invertMatrixT(u + applyAT(y) - (1 / t) * (w + applyAT(z)))
        return xSol

    def should_use(self, algorithm: str) -> bool:
        return algorithm == ALGORITHM_ADMM
