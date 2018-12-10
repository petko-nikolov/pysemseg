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


class ResizeBatch:
    """
    params:
        size: (height, width)
    returns:
        resized image
    """
    def __init__(self, size, interpolation=cv2.INTER_LANCZOS4):
        self.size = size
        self.interpolation = interpolation
        self.resize_op = Resize(size, interpolation=interpolation)

    def __call__(self, images):
        resized_images = []
        for i in range(images.shape[0]):
            resized_images.append(self.resize_op(images[i]))
        return np.stack(resized_images)


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


class RandomGammaCorrection:
    def __init__(self, min_gamma=0.75, max_gamma=1.25):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, image):
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        image = cv2.LUT(image, table)
        return image / 255.

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


class RandomRotate:
    def __init__(self, max_delta=5.0):
        self.max_delta = max_delta

    def __call__(self, image, mask):
        angle = np.random.uniform(-self.max_delta, self.max_delta)
        image = self._rotate(image, angle)
        mask = self._rotate(mask, angle, interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _rotate(self, image, angle, interpolation=cv2.INTER_LANCZOS4):
        image_center = (image.shape[1] / 2, image.shape[0] / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, (image.shape[1], image.shape[0]), flags=interpolation,
            borderMode=cv2.BORDER_REPLICATE)
        return result


class RandomTranslate:
    def __init__(self, max_percent_delta=0.02):
        self.max_percent_delta = max_percent_delta

    def __call__(self, image, mask):
        delta_x = np.random.uniform(-self.max_percent_delta, self.max_percent_delta)
        delta_y = np.random.uniform(-self.max_percent_delta, self.max_percent_delta)
        image = self._translate(image, delta_x, delta_y)
        mask = self._translate(mask, delta_x, delta_y)
        return image, mask

    def _translate(self, image, delta_x, delta_y):
        height, width = image.shape[:2]
        translation_matrix = np.float32([
            [1, 0, int(width * delta_x)],
            [0, 1, int(height * delta_y)]
        ])
        translated_image = cv2.warpAffine(
            image, translation_matrix, (width, height),
            borderMode=cv2.BORDER_REPLICATE
        )
        return translated_image


class RandomCropFixedSize:
    def __init__(self, size):
        self.height, self.width = size

    def __call__(self, image, mask):
        assert image.shape[:2] == mask.shape[:2]
        assert image.shape[0] >= self.height and image.shape[1] >= self.width
        scol = np.random.randint(0, image.shape[0] - self.height + 1)
        srow = np.random.randint(0, image.shape[1] - self.width + 1)
        image_crop = image[scol:scol+self.height, srow:srow+self.width]
        mask_crop = mask[scol:scol+self.height, srow:srow+self.width]
        return image_crop, mask_crop


class RandomCrop:
    def __init__(self, scale_height=(0.8, 1.0), scale_width=(0.8, 1.0)):
        self.scale_height = scale_height
        self.scale_width = scale_width

    def __call__(self, image, mask):
        assert image.shape[:2] == mask.shape[:2]
        sh = np.random.uniform(*self.scale_height)
        sw = np.random.uniform(*self.scale_width)
        crop_height = int(sh * image.shape[0])
        crop_width = int(sw * image.shape[1])
        return RandomCropFixedSize((crop_height, crop_width))(image, mask)


class RandomScale:
    def __init__(self, scale_height=(0.8, 1.2), scale_width=(0.8, 1.2),
                 interpolation=cv2.INTER_LANCZOS4):
        self.scale_height = scale_height
        self.scale_width = scale_width

    def __call__(self, image, mask):
        height = int(image.shape[0] * np.random.uniform(*self.scale_height))
        width = int(image.shape[1] * np.random.uniform(*self.scale_width))
        return (
            Resize((height, width))(image),
            Resize((height, width), cv2.INTER_NEAREST)(mask)
        )


class ScaleTo:
    def __init__(self, size, interpolation=cv2.INTER_LANCZOS4):
        self.height, self.width = size
        self.interpolation = interpolation

    def __call__(self, image, mask):
        scale = min(self.height / image.shape[0], self.width / image.shape[1])
        height = int(scale * image.shape[0])
        width = int(scale * image.shape[1])
        return (
            cv2.resize(image, (width, height),
                interpolation=self.interpolation),
            cv2.resize(mask, (width, height),
                interpolation=cv2.INTER_NEAREST)
        )


class Concat:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        assert len(args) == len(self.transforms)
        return [
            tr(x) for tr, x in zip(self.transforms, args)
        ]

class PadTo:
    def __init__(self, size, pad_value=0):
        self.height, self.width = size
        self.pad_value = pad_value

    def __call__(self, image):
        hpad = max(0, self.height - image.shape[0])
        wpad = max(0, self.width - image.shape[1])
        top_pad = hpad // 2
        bottom_pad = hpad // 2 + hpad % 2
        left_pad = wpad // 2
        right_pad = wpad // 2 + wpad % 2
        image = cv2.copyMakeBorder(
            image, top_pad, bottom_pad, left_pad,
            right_pad, cv2.BORDER_CONSTANT, value=self.pad_value
        )
        return image
