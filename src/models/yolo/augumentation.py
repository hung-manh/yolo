import cv2
import numpy as np
from typing import Tuple    


class YoloAugmentation:
    def __init__(self, image: np.ndarray) -> None: 
        """
        Initialize the YoloAugmentation class with an image.

        Parameters:
        - image (numpy.ndarray): Input image on which augmentations will be applied.
        """
        self.image = image

    def flip(self, flip_code:int=1) -> np.ndarray:
        """
        Flip the image horizontally, vertically, or both.

        Parameters:
        - flip_code (int): Determines the flipping direction. 
          1 = horizontal flip, 0 = vertical flip, -1 = both.

        Returns:
        - numpy.ndarray: Flipped image.
        """
        return cv2.flip(self.image, flip_code)

    def rotate(self, angle: float=45) -> np.ndarray:
        """
        Rotate the image by a specified angle.

        Parameters:
        - angle (float): Angle in degrees to rotate the image.

        Returns:
        - numpy.ndarray: Rotated image.
        """
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)  # Define the center for rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotation matrix
        rotated_image = cv2.warpAffine(self.image, M, (w, h))
        return rotated_image

    def scale(self, scale_percent: int=120) -> np.ndarray:
        """
        Scale the image by a specified percentage.

        Parameters:
        - scale_percent (int): Percentage to scale the image size.

        Returns:
        - numpy.ndarray: Scaled image.
        """
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        scaled_image = cv2.resize(self.image, (width, height))
        return scaled_image

    def adjust_brightness_contrast(self, brightness=50, contrast=1.5) -> np.ndarray:
        """
        Adjust the brightness and contrast of the image.

        Parameters:
        - brightness (int): Value to increase or decrease brightness.
        - contrast (float): Multiplier for contrast adjustment.

        Returns:
        - numpy.ndarray: Image with adjusted brightness and contrast.
        """
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=brightness)
        return adjusted_image

    def add_noise(self, mean=0, sigma=25) -> np.ndarray:
        """
        Add Gaussian noise to the image.

        Parameters:
        - mean (float): Mean of the Gaussian noise distribution.
        - sigma (float): Standard deviation of the Gaussian noise.

        Returns:
        - numpy.ndarray: Noisy image.
        """
        noise = np.random.normal(mean, sigma, self.image.shape).astype(np.uint8)
        noisy_image = cv2.add(self.image, noise)
        return noisy_image

    def blur(self, kernel_size:Tuple[int, int]=(5, 5)) -> np.ndarray:
        """
        Apply Gaussian blur to the image.

        Parameters:
        - kernel_size (tuple): Kernel size for the Gaussian blur.

        Returns:
        - numpy.ndarray: Blurred image.
        """
        blurred_image = cv2.GaussianBlur(self.image, kernel_size, 0)
        return blurred_image

    def translate(self, tx:int=50, ty:int=30) -> np.ndarray:
        """
        Translate the image horizontally and/or vertically.

        Parameters:
        - tx (int): Pixels to shift along the x-axis.
        - ty (int): Pixels to shift along the y-axis.

        Returns:
        - numpy.ndarray: Translated image.
        """
        (h, w) = self.image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix
        translated_image = cv2.warpAffine(self.image, M, (w, h))
        return translated_image

    def color_jitter(self, hue_shift:int=10, saturation_shift:int=10) -> np.ndarray:
        """
        Adjust hue and saturation to create color jitter effects.

        Parameters:
        - hue_shift (int): Value to adjust hue.
        - saturation_shift (int): Value to adjust saturation.

        Returns:
        - numpy.ndarray: Image with adjusted hue and saturation.
        """
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        h = cv2.add(h, hue_shift)
        s = cv2.add(s, saturation_shift)
        jittered_image = cv2.merge([h, s, v])
        jittered_image = cv2.cvtColor(jittered_image, cv2.COLOR_HSV2BGR)
        return jittered_image
