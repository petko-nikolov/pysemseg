from PIL import Image
import cv2


class PILImageLoader():
    def __call__(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class CV2ImageLoader():
    def __init__(self, grayscale=False):
        self.grayscale = grayscale

    def __call__(self, path):
        if self.grayscale:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
