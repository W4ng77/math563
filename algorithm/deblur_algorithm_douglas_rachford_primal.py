from algorithm.abstract_deblur_algorithm import AbstractDeblurAlgorithm
from prox.prox_calculator_provider import ProxCalculatorProvider
from utils.constants import *
import numpy as np

class DeblurAlgorithmDouglasRachfordPrimal(AbstractDeblurAlgorithm):
    def run_algorithm(self, **kwargs):
        b = kwargs["b"]  # blurred image
        t = kwargs["tprimaldr"]  # step size
        rho = kwargs["rhoprimaldr"]  # relaxation parameter
        max_iter = kwargs.get("max_iter", 100)
        norm_type = kwargs["norm_type"]
        gamma = kwargs["gammal1"]

        provider = ProxCalculatorProvider()

        # load prox operators
        prox_f = provider.provide_calculator(BOX_PROX)
        prox_f.t = t

        # Fixed: Pass the parameters directly to provide_calculator
        prox_g = provider.provide_calculator(
            G_PROX,
            norm_type=norm_type,
            b=b,
            t=t,
            gamma=gamma
        )

        shape = b.shape
        z1 = np.zeros(shape)         # for x
        z2 = np.zeros((*shape, 3))   # for y = (y1, y2, y3)

        K = kwargs["K"]      # forward operator (blurring)
        D = kwargs["D"]      # gradient operator
        At = kwargs["At"]    # adjoint of [K; D]
        solve_fft = kwargs["solve_fft"]  # FFT-based solver for (I + A^T A)^(-1)

        def A(x):
            kx = K(x)
            dx1, dx2 = D(x)
            return np.stack([kx, dx1, dx2], axis=-1)

        for _ in range(max_iter):
            xk = prox_f.calculate(z1)
            yk = prox_g.calculate(z2)

            rhs = 2 * xk - z1 + At(2 * yk - z2)
            uk = solve_fft(rhs)
            vk = A(uk)

            z1 += rho * (uk - xk)
            z2 += rho * (vk - yk)

        return prox_f.calculate(z1)

    def should_use(self, algorithm: str) -> bool:
        return algorithm == ALGORITHM_DOUGLAS_RACHFORD_PRIMAL