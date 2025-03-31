from problem.abstract_deblur_problem import AbstractDeblurProblem
from utils.constants import *
import numpy as np

class DeblurProblemL2(AbstractDeblurProblem):
    def run_problem(self, **kwargs):
        b = kwargs.get("b")
        kernel = kwargs.get("kernel")
        gamma = kwargs.get("gamma", 0.1)
        alpha = kwargs.get("alpha", 1.0)  # weight for L2 fidelity term

        def prox_f(v):
            # Proximal operator for f = ||Kx - b||_2^2 is quadratic + indicator
            return np.clip(v, 0, 1)  # projection onto [0, 1]

        def prox_g(y):
            # Isotropic TV projection
            norm = np.maximum(1, np.sqrt(np.sum(y**2, axis=0)))
            return y / norm

        return prox_f, prox_g, gamma

    def should_use(self, problem: str) -> bool:
        return problem == L2_PROBLEM
