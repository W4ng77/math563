import numpy as np
from scipy.signal import convolve2d


def apply_periodic_conv2d(x: np.ndarray, eigval_arr: np.ndarray) -> np.ndarray:
    """
    Applies 2D circular convolution in the frequency domain.
    This simulates convolution via multiplication in Fourier space.

    Parameters:
        x (np.ndarray): Input image
        eigval_arr (np.ndarray): Eigenvalue array of the convolution kernel (DFT)

    Returns:
        np.ndarray: Blurred image
    """
    return np.real(np.fft.ifft2(np.fft.fft2(x) * eigval_arr))


def compute_eigenvalues(kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Computes the eigenvalue array (Fourier transform) of a convolution kernel
    with periodic boundary conditions.

    Parameters:
        kernel (np.ndarray): Convolution kernel (e.g., Gaussian blur kernel)
        shape (tuple): Target shape (height, width) for zero-padding

    Returns:
        np.ndarray: Eigenvalue array (FFT of padded kernel)
    """
    padded_kernel = np.zeros(shape)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -kh//2, axis=0)
    padded_kernel = np.roll(padded_kernel, -kw//2, axis=1)

    return np.fft.fft2(padded_kernel)
