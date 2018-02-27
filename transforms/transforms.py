import cv2
import numpy as np


class Resize:
    """
    params:
        size: (height, width)
    returns:
        resized image
    """
    def __init__(self, size, interpolation=cv2.INTER_LANCZOS4):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, ):
        return cv2.resize(
            image, self.size[::-1], interpolation=self.interpolation)


class Binarize:
    """
    params:
        threshold: binary threshold
    returns:
        binarized image
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, image):
        _, binarized_image = cv2.threshold(
            image, self.threshold, 255, cv2.THRESH_BINARY)
        return binarized_image


class RandomContrast:
    """
    params:
        contrast level:
    returns:
        image with changed contrast
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        assert image.dtype == np.float
        contrast = np.random.uniform(self.low, self.high)
        image = image * contrast
        return np.clip(image, 0.0, 1.0)


class RandomBrightness:
    """
    params:
        brightness level
    returns:
        image with changed birghtness
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        assert image.dtype == np.float
        brightness = np.random.uniform(self.low, self.high)
        image = image + brightness
        return np.clip(image, 0.0, 1.0)


class RandomHueSaturation:
    """
    params:
        saturation adjustment
    returns:
        saturated image
    """
    def __init__(self, hue_delta_range, saturation_delta_range):
        self.hue_delta_range = hue_delta_range
        self.saturation_delta_range = saturation_delta_range

    def _adjust_hue(self, hue, delta):
        hue += delta * 360
        hue[hue < 0] += 360
        hue[hue >= 360] -= 360
        return hue

    def __call__(self, image):
        assert image.dtype == np.float
        image = image.astype(np.float32)
        saturation = np.random.uniform(*self.saturation_delta_range)
        hue = np.random.uniform(*self.hue_delta_range)
        hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(hsvimage)
        s = np.clip(s + s * saturation, 0.0, 1.0)
        h = self._adjust_hue(h, hue)
        hsvimage = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsvimage, cv2.COLOR_HSV2BGR)
        return image
