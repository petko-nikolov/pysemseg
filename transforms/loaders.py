from PIL import Image
import cv2


class PILImageLoader():
    def __call__(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class CV2ImageLoader():
    def __call__(self, path):
        return cv2.imread(path)

