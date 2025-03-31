import numpy as np
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2

def gradient(x):
    """
    Compute forward finite differences (Dx, Dy)
    """
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    return dx, dy

def divergence(dx, dy):
    """
    Compute divergence (i.e., adjoint of gradient operator)
    """
    ddx = dx - np.roll(dx, 1, axis=1)
    ddy = dy - np.roll(dy, 1, axis=0)
    return ddx + ddy

def apply_blur(x, kernel):
    return convolve2d(x, kernel, mode='same', boundary='wrap')

def apply_blur_transpose(x, kernel):
    return convolve2d(x, np.flipud(np.fliplr(kernel)), mode='same', boundary='wrap')

def make_applyA(kernel):
    def applyA(x):
        kx = apply_blur(x, kernel)
        dx, dy = gradient(x)
        return np.stack([kx, dx, dy], axis=-1)
    return applyA

def make_applyAT(kernel):
    def applyAT(y):
        y1, y2, y3 = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        kt = apply_blur_transpose(y1, kernel)
        div = divergence(y2, y3)
        return kt + div
    return applyAT

def make_invert_matrix_t(kernel, t):
    def compute_fft_kernel(h, w):
        fft_K = fft2(kernel, s=(h, w))
        K_mag = np.abs(fft_K) ** 2
        fx = np.zeros((h, w))
        fx[0, 0] = -1
        fx[0, 1] = 1
        fy = np.zeros((h, w))
        fy[0, 0] = -1
        fy[1, 0] = 1
        fft_fx = fft2(fx)
        fft_fy = fft2(fy)
        D_mag = np.abs(fft_fx) ** 2 + np.abs(fft_fy) ** 2
        return 1 / (1 + t**2 * (K_mag + D_mag))

    def invertMatrixT(x):
        h, w = x.shape
        if not hasattr(invertMatrixT, "cached_filter") or invertMatrixT.cached_filter.shape != (h, w):
            invertMatrixT.cached_filter = compute_fft_kernel(h, w)
        return np.real(ifft2(fft2(x) * invertMatrixT.cached_filter))

    return invertMatrixT
