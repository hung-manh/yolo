import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any


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


class DataAugmentation:
    def __init__(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the DataAugmentation class with an image and optional augmentation parameters.

        Parameters:
        - image (numpy.ndarray): Input image on which augmentations will be applied.
        - params (dict, optional): Dictionary of augmentation parameters.
            Supported keys:
            - hsv_h (float): Hue shift factor
            - hsv_s (float): Saturation shift factor
            - hsv_v (float): Value (brightness) shift factor
            - degrees (float): Rotation angle
            - translate (float): Translation factor
            - scale (float): Scaling factor
            - shear (float): Shear factor
            - perspective (float): Perspective transformation factor
            - flipud (float): Probability of upside-down flip
            - fliplr (float): Probability of left-right flip
            - mosaic (float): Mosaic augmentation factor
            - erasing (float): Random erasing factor
        """
        self.image = image
        self.params = params or {}
        
        # Default augmentation parameters
        self.default_params = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 1.0,
            'erasing': 0.4
        }
        
        # Merge default and user-provided parameters
        self.config = {**self.default_params, **self.params}

    def _random_affine(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random affine transformations based on configuration.

        Parameters:
        - image (numpy.ndarray): Input image

        Returns:
        - numpy.ndarray: Transformed image
        """
        height, width = image.shape[:2]
        
        # Random rotation
        angle = self.config['degrees'] * np.random.uniform(-1, 1)
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Random translation
        translate_x = int(width * self.config['translate'] * np.random.uniform(-1, 1))
        translate_y = int(height * self.config['translate'] * np.random.uniform(-1, 1))
        
        # Random scaling
        scale = 1 + self.config['scale'] * np.random.uniform(-1, 1)
        
        # Combine transformations
        matrix = rotation_matrix.copy()
        matrix[0, 2] += translate_x
        matrix[1, 2] += translate_y
        
        # Apply scaling
        matrix[0, 0] *= scale
        matrix[1, 1] *= scale
        
        transformed_image = cv2.warpAffine(
            image, 
            matrix, 
            (width, height), 
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return transformed_image

    def _random_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random HSV color jittering.

        Parameters:
        - image (numpy.ndarray): Input image

        Returns:
        - numpy.ndarray: HSV augmented image
        """
        hsv_image = image.copy().astype(np.float32) / 255.0
        hsv = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
        
        # Random HSV augmentation
        h_scale = 1 + np.random.uniform(-1, 1) * self.config['hsv_h']
        s_scale = 1 + np.random.uniform(-1, 1) * self.config['hsv_s']
        v_scale = 1 + np.random.uniform(-1, 1) * self.config['hsv_v']
        
        hsv[:, :, 0] *= h_scale
        hsv[:, :, 1] *= s_scale
        hsv[:, :, 2] *= v_scale
        
        # Clip values
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 1)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 1)
        
        augmented_image = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return augmented_image

    def _random_flip(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random horizontal and vertical flips.

        Parameters:
        - image (numpy.ndarray): Input image

        Returns:
        - numpy.ndarray: Flipped image
        """
        if np.random.random() < self.config['fliplr']:
            image = cv2.flip(image, 1)
        
        if np.random.random() < self.config['flipud']:
            image = cv2.flip(image, 0)
        
        return image

    def _random_erase(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random erasing to the image.

        Parameters:
        - image (numpy.ndarray): Input image

        Returns:
        - numpy.ndarray: Image with random region erased
        """
        if np.random.random() < self.config['erasing']:
            height, width = image.shape[:2]
            erase_area = int(min(height, width) * 0.2)  # Erase 20% of minimum dimension
            
            x = np.random.randint(0, width - erase_area)
            y = np.random.randint(0, height - erase_area)
            
            image[y:y+erase_area, x:x+erase_area] = np.random.randint(0, 255, (erase_area, erase_area, 3))
        
        return image

    def augment(self) -> np.ndarray:
        """
        Apply a series of augmentations to the image based on configured parameters.

        Returns:
        - numpy.ndarray: Augmented image
        """
        augmented_image = self.image.copy()
        
        # Apply augmentations in order
        augmented_image = self._random_affine(augmented_image)
        augmented_image = self._random_hsv(augmented_image)
        augmented_image = self._random_flip(augmented_image)
        augmented_image = self._random_erase(augmented_image)
        
        return augmented_image