import cv2
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for transform in self.transforms:
            if args:
                image, *args = transform(image, *args)
            else:
                image = transform(image)
        if args:
            return (image,) + tuple(args)
        else:
            return image

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

    def __call__(self, image):
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
        assert np.issubdtype(image.dtype, np.floating)
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
        assert np.issubdtype(image.dtype, np.floating)
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
    def __init__(self, hue_delta=0.1, saturation_scale_range=(0.7, 1.3)):
        self.hue_delta = hue_delta
        self.saturation_scale_range = saturation_scale_range

    def _adjust_hue(self, hue, delta):
        hue += delta * 360
        hue[hue < 0] += 360
        hue[hue >= 360] -= 360
        return hue

    def __call__(self, image):
        assert np.issubdtype(image.dtype, np.floating)
        image = image.astype(np.float32)
        saturation = np.random.uniform(*self.saturation_scale_range)
        hue = np.random.uniform(-self.hue_delta, self.hue_delta)
        hsvimage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        (h, s, v) = cv2.split(hsvimage)
        s = np.clip(s * saturation, 0.0, 1.0)
        h = self._adjust_hue(h, hue)
        hsvimage = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsvimage, cv2.COLOR_HSV2RGB)
        return image


class RandomHorizontalFlip:
    """
    params:
        Probability to flip the image horizontally
    returns:
        Maybe flipped image
    """
    def __init__(self, flip_probability=0.5):
        self.flip_probability = flip_probability

    def __call__(self, image, mask):
        if np.random.random() < self.flip_probability:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask
