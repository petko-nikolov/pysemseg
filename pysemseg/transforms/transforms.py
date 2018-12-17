import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter



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
        dims = len(image.shape)
        image = cv2.resize(
            image, self.size[::-1], interpolation=self.interpolation)
        if len(image.shape) == dims -1:
            image = np.expand_dims(image, -1)
        return image



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
        image = np.clip(image, 0.0, 1.0)
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
        image = image / 255.
        image = np.clip(image, 0.0, 1.0)
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


class RandomRotate:
    def __init__(self, max_delta=5.0):
        self.max_delta = max_delta

    def __call__(self, image, mask):
        angle = np.random.uniform(-self.max_delta, self.max_delta)
        image = self._rotate(image, angle, interpolation=cv2.INTER_LINEAR)
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
    def __init__(self, scale_height=(0.8, 1.2), scale_width=(0.8, 1.2), aspect_range=[0.8, 1.2]):
        self.scale_height = scale_height
        self.scale_width = scale_width
        self.aspect_range = aspect_range

    def __call__(self, image, mask):
        hs = np.random.uniform(*self.scale_height)
        height = int(image.shape[0] * hs)
        lw = max(hs * self.aspect_range[0], self.scale_width[0])
        hw = min(hs * self.aspect_range[1], self.scale_width[1])
        ws = np.random.uniform(lw, hw)
        width = int(image.shape[1] * ws)
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


class Choice:
    def __init__(self, choices, p=None):
        self.choices = choices
        self.p = p
        if not self.p:
            self.p = np.ones(len(self.choices)) / len(self.choices)

    def __call__(self, *args):
        c = np.random.choice(self.choices, p=self.p)
        return c(*args)


class RandomPerspective:
    def __init__(self, corner_offset=0.15):
        self.corner_offset = corner_offset

    def __call__(self, image, mask):
        height, width = image.shape[:2]
        lh = int(self.corner_offset * height)
        hh = int(height - self.corner_offset * height)
        lw = int(self.corner_offset * width)
        hw = int(width - self.corner_offset * width)
        x1, x3 = np.random.randint(0, lw, [2])
        y1, y2 = np.random.randint(0, lh, [2])
        x2, x4 = np.random.randint(hw, width, [2])
        y3, y4 = np.random.randint(hh, height, [2])
        mapped = np.float32([
             [x1, y1],
             [x2, y2],
             [x3, y3],
             [x4, y4]
        ])
        corners = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])
        M = cv2.getPerspectiveTransform(mapped, corners)
        image = cv2.warpPerspective(
            image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpPerspective(
            mask, M, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT)

        return image, mask

class RandomShear:
    def __init__(self, max_range=0.01):
        self.max_range = max_range

    def __call__(self, image, mask):
        height, width = image.shape[:2]
        cx = width / 2
        cy = height / 2
        pts = np.float32([
            [cx, cy],
            [cx + 0.1 * width, cy],
            [cx, cy + 0.1 * height]
        ])
        ru = lambda x: x * np.random.uniform(
            -self.max_range, self.max_range)

        mapped = np.float32([
            [pts[0][0] + ru(width), pts[0][1]],
            [pts[1][0] + ru(width), pts[1][1] + ru(height)],
            [pts[2][0], pts[2][1] + ru(height)]
        ])

        M = cv2.getAffineTransform(pts, mapped)

        image = cv2.warpAffine(
            image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(
            mask, M, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        return image, mask


class RandomElasticTransform:
    """
    Adapted from
    https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

    """
    def __init__(self, sigma=(0.07, 0.15), alpha=(3, 3.5)):
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, image, mask):
        height, width = image.shape[:2]
        sigma = np.random.uniform(*self.sigma)
        x_sigma, y_sigma = sigma * width, sigma * height
        alpha = np.random.uniform(*self.alpha)
        x_alpha, y_alpha = alpha * width, alpha * height
        x_, y_ = np.meshgrid(np.arange(0, width), np.arange(0, height), indexing='xy')
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), x_sigma) * x_alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), y_sigma) * y_alpha
        x = (x_ + dx).astype(np.float32)
        y = (y_ + dy).astype(np.float32)
        image = cv2.remap(
            image, x, y, interpolation=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REFLECT)
        mask = cv2.remap(
            mask, x, y, interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT)
        return image, mask


class RandomGaussianBlur:
    def __init__(self, kernel_sizes=(3, 5, 7), apply_prob=0.2):
        self.kernel_sizes = kernel_sizes
        self.apply_prob = apply_prob

    def __call__(self, image):
        if np.random.rand() < self.apply_prob:
            ksize = np.random.choice(self.kernel_sizes)
            image = cv2.GaussianBlur(image, (ksize,) * 2, 0)
        return image
