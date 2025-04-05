from algorithm.deblur_algorithm_factory import get_algorithm
import numpy as np
from scipy.signal import convolve2d
from skimage.util import random_noise


def gaussian_kernel(size=15, sigma=5):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def optsolve(I, cfg):
    # 1. Create Gaussian kernel
    kernel = gaussian_kernel(size=15, sigma=cfg.experiment.sigma)

    # 2. Blur + noise
    b = convolve2d(I, kernel, mode='same', boundary='wrap')
    b = random_noise(b, mode='s&p', amount=cfg.experiment.noise_density)

    # 3. Construct eigenvalues and operators
    fft_shape = I.shape
    eig_K = np.fft.fft2(kernel, s=fft_shape)
    eig_KT = np.conj(eig_K)

    eig_D1 = np.fft.fft2([[-1, 1]], s=fft_shape)
    eig_D2 = np.fft.fft2([[-1], [1]], s=fft_shape)
    eig_D1T = np.conj(eig_D1)
    eig_D2T = np.conj(eig_D2)

    eig_all = (1 + cfg.experiment.tadmm**2 * (np.abs(eig_KT * eig_K) +
                                               np.abs(eig_D1T * eig_D1) +
                                               np.abs(eig_D2T * eig_D2)))

    def applyK(x): return np.real(np.fft.ifft2(eig_K * np.fft.fft2(x)))
    def applyKT(x): return np.real(np.fft.ifft2(eig_KT * np.fft.fft2(x)))
    def applyD(x): return np.stack([
        np.real(np.fft.ifft2(eig_D1 * np.fft.fft2(x))),
        np.real(np.fft.ifft2(eig_D2 * np.fft.fft2(x)))
    ], axis=-1)
    def applyDT(y): return np.real(np.fft.ifft2(eig_D1T * np.fft.fft2(y[..., 0]))) + \
                            np.real(np.fft.ifft2(eig_D2T * np.fft.fft2(y[..., 1])))

    applyA = lambda x: np.concatenate([applyK(x)[..., np.newaxis], applyD(x)], axis=-1)
    applyAT = lambda y: applyKT(y[..., 0]) + applyDT(y[..., 1:])
    invertMatrixT = lambda x: np.real(np.fft.ifft2(np.fft.fft2(x) / eig_all))

    # 4. Init vectors
    algorithm = cfg.experiment.algorithm
    if algorithm == "chambollepock":
        init_vectors = [
            np.zeros_like(b),
            np.zeros((*b.shape, 3)),
            np.zeros_like(b)
        ]
    else:
        init_vectors = [
            np.zeros_like(b),
            np.zeros((*b.shape, 3)),
            np.zeros_like(b),
            np.zeros((*b.shape, 3))
        ]

    # 5. Params
    params = {
        "tadmm": cfg.experiment.tadmm,
        "rhoadmm": cfg.experiment.rhoadmm,
        "gammal1": cfg.experiment.gammal1,
        "gammal2": cfg.experiment.get("gammal2", 0.0),
        "maxiter": cfg.experiment.maxiter,
        "tchambollepock": cfg.experiment.get("tchambollepock", 0.5),
        "schambollepock": cfg.experiment.get("schambollepock", 0.5),
        "tprimaldr": cfg.experiment.get("tprimaldr", 2.0),
        "rhoprimaldr": cfg.experiment.get("rhoprimaldr", 0.1),
        "tprimaldualdr": cfg.experiment.get("tprimaldualdr", 2.0),
        "rhoprimaldualdr": cfg.experiment.get("rhoprimaldualdr", 1.049)
    }

    # 6. Run algorithm
    algo = get_algorithm(algorithm)
    x_restored = algo.run_algorithm(
        b=b,
        init_vectors=init_vectors,
        problem=cfg.experiment.problem,
        params=params,
        applyA=applyA,
        applyAT=applyAT,
        invertMatrixT=invertMatrixT,
        norm_type=cfg.experiment.problem,
        t=1 / params["schambollepock"] if algorithm == "chambollepock" else 1 / params["tadmm"],
        gamma=params["gammal1"] if cfg.experiment.problem == "l1" else params["gammal2"]
    )

    return b, x_restored
