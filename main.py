import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.util import random_noise
import matplotlib.pyplot as plt


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
    b = gaussian_filter(I, sigma=cfg.experiment.sigma)

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

    # Run deblurring algorithm
    algo = get_algorithm(cfg.algorithm.name, cfg)
    restored = algo.run_algorithm(b=b, kernel=kernel)

    # Display restored image
    plt.figure('Restored Image')
    plt.imshow(restored, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
