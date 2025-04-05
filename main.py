import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.util import random_noise
import matplotlib.pyplot as plt
from algorithm.deblur_algorithm_factory import get_algorithm
from utils.constants import ALGORITHM_DOUGLAS_RACHFORD_PRIMAL


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load image
    I = cv2.imread(cfg.experiment.image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Convert to double (float) and normalize
    I = img_as_float(I)
    I = I - np.min(I)  # Normalize min to 0
    I = I / np.max(I)  # Normalize max to 1

    # Display image before blurring
    plt.figure('Image before blurring')
    plt.imshow(I, cmap='gray')
    plt.axis('off')
    plt.show()

    # Gaussian filter
    sigma = cfg.experiment.sigma
    b = gaussian_filter(I, sigma=sigma)

    # Add salt and pepper noise
    b = random_noise(
        b,
        mode='s&p',
        amount=cfg.experiment.noise_density)

    # Display blurred and noisy image
    plt.figure('Blurred and Noisy Image')
    plt.imshow(b, cmap='gray')
    plt.axis('off')
    plt.show()

    # Create gaussian kernel for PSF
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    x = np.arange(-(kernel_size//2), kernel_size//2 + 1)
    y = np.arange(-(kernel_size//2), kernel_size//2 + 1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2)/(2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Setup FFT-based operators
    def K(x):
        # Convolution operator using FFT
        return cv2.filter2D(x, -1, kernel)
        
    def D(x):
        # Gradient operator
        dx1 = np.zeros_like(x)
        dx2 = np.zeros_like(x)
        dx1[:-1, :] = x[1:, :] - x[:-1, :]  # Vertical gradient
        dx2[:, :-1] = x[:, 1:] - x[:, :-1]  # Horizontal gradient
        return dx1, dx2
        
    def At(y):
        # Adjoint operator of [K; D]
        if y.ndim == 3:  # Handling stacked output from A
            ky = cv2.filter2D(y[:, :, 0], -1, np.flipud(np.fliplr(kernel)))
            d1y = np.zeros_like(y[:, :, 1])
            d2y = np.zeros_like(y[:, :, 2])
            
            # Adjoint of vertical gradient
            d1y[1:, :] = y[:-1, :, 1]
            d1y[0, :] = 0
            d1y = -d1y
            
            # Adjoint of horizontal gradient
            d2y[:, 1:] = y[:, :-1, 2]
            d2y[:, 0] = 0
            d2y = -d2y
            
            return ky + d1y + d2y
        else:
            # Just K adjoint
            return cv2.filter2D(y, -1, np.flipud(np.fliplr(kernel)))
    
    # FFT-based solver for (I + A^T A)^(-1)
    def solve_fft(rhs):
        # Simple implementation - in practice you would use FFT-based solvers
        # This is a placeholder that assumes a simple denoising model
        return rhs  # Replace with proper solver
    
    # Get the algorithm
    algo = get_algorithm(cfg.experiment.algorithm)
    
    if algo is None:
        raise ValueError(f"Algorithm {cfg.experiment.algorithm} not found!")
    
    # Prepare parameters
    params = {
        "tprimaldr": cfg.experiment.tprimaldr,
        "rhoprimaldr": cfg.experiment.rhoprimaldr,
        "gammal1": cfg.experiment.gammal1,
        "max_iter": cfg.experiment.maxiter,
        "norm_type": cfg.experiment.problem
    }
    
    # Run deblurring algorithm
    restored = algo.run_algorithm(
        b=b,                # blurred image
        K=K,                # forward operator
        D=D,                # gradient operator
        At=At,              # adjoint operator
        solve_fft=solve_fft,# FFT solver
        gammal1=cfg.experiment.gammal1,
        tprimaldr=cfg.experiment.tprimaldr,
        rhoprimaldr=cfg.experiment.rhoprimaldr,
        norm_type=cfg.experiment.problem,
        max_iter=cfg.experiment.maxiter
    )

    # Display restored image
    plt.figure('Restored Image')
    plt.imshow(restored, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()