import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import cv2
from skimage import img_as_float
import matplotlib.pyplot as plt
from solver.optsolve import optsolve


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load image
    I = cv2.imread(cfg.experiment.image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    I = img_as_float(I)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))  # Normalize to [0,1]

    # Display image before blurring
    plt.figure('Original Image')
    plt.imshow(I, cmap='gray')
    plt.axis('off')
    plt.show()

    # Run deblurring and get noisy image and restored result
    b, restored = optsolve(I, cfg)

    # Display blurred and noisy image
    plt.figure('Blurred and Noisy Image')
    plt.imshow(b, cmap='gray')
    plt.axis('off')
    plt.show()

    # Display restored image
    plt.figure('Restored Image')
    plt.imshow(restored, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
