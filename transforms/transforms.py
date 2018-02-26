import cv2


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
