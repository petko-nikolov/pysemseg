import cv2


class Resize:
    """
    params:
        size: (height, width)
    returns:
        resized image
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, interpolation=cv2.INTER_LANCZOS4):
        return cv2.resize(
            image, self.size[::-1], interpolation=interpolation)
