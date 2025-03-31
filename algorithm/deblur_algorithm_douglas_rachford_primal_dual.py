from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm
from prox.prox_calculator_provider import ProxCalculatorProvider
from utils.constants import *
import numpy as np

class DeblurAlgorithmDouglasRachfordPrimalDual(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        # === Required Inputs ===
        b = kwargs["b"]  # blurred image
        init_vectors = kwargs["init_vectors"]  # [p, q]
        problem = kwargs["problem"]  # "l1" or "l2"
        i = kwargs["params"]  # param dict: t, rho, gamma, maxiter
        applyA = kwargs["applyA"]     # function: apply A(x)
        applyAT = kwargs["applyAT"]   # function: apply A^T(x)
        invertMatrixT = kwargs["invertMatrixT"]  # function: inverse via FFT

        # === Parameters ===
        t = i["tprimaldualdr"]
        rho = i["rhoprimaldualdr"]
        gamma = i["gammal1"]
        max_iter = i["maxiter"]

        # === Initial Vectors ===
        p = init_vectors[0]
        q = init_vectors[1]

        # === Load Prox Operators ===
        provider = ProxCalculatorProvider()
        prox_f = provider.provide_calculator(BOX_PROX)
        prox_g = provider.provide_calculator(G_PROX)

        # === Set Prox Operator Parameters ===
        prox_f.t = t
        prox_g.t = 1 / t
        prox_g.b = b
        prox_g.gamma = gamma
        prox_g.norm_type = problem

        # === Iteration ===
        for _ in range(max_iter):
            x = prox_f.calculate(p)
            z = q - t * prox_g.calculate(q / t)  # Moreau decomposition

            matrix_one = 2 * x - p
            matrix_two = 2 * z - q

            # w = matrix_one - t^2 A^T (inv(A A^T) A matrix_one) - t A^T inv matrix_two
            w = matrix_one - t**2 * applyAT(invertMatrixT(applyA(matrix_one))) - t * applyAT(invertMatrixT(matrix_two))
            v = t * invertMatrixT(applyA(matrix_one)) + invertMatrixT(matrix_two)

            p = p + rho * (w - x)
            q = q + rho * (v - z)

        # Final projection
        x_final = prox_f.calculate(p)
        return x_final

    def should_use(self, algorithm: str) -> bool:
        return algorithm == ALGORITHM_DOUGLAS_RACHFORD_PRIMAL_DUAL
