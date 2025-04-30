import random
from io import BytesIO
import kornia.feature as KF
import kornia as K
from misc.util import top_n_to_one
from torchdistill.common.module_util import freeze_module_params
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import RandomResizedCrop, Resize
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from ultralytics.data.augment import LetterBox

from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)

TRANSFORM_CLASS_DICT = dict()
INTERPOLATION_MODE_DICT = {
    "nearest": InterpolationMode.NEAREST,
    "bicubic": InterpolationMode.BICUBIC,
    "bilinear": InterpolationMode.BILINEAR,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS,
}


def register_transform_class(cls):
    TRANSFORM_CLASS_DICT[cls.__name__] = cls
    return cls


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


@register_transform_class
class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


@register_transform_class
class CustomRandomResize(object):
    def __init__(self, min_size, max_size=None, jpeg_quality=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size

        self.max_size = max_size
        self.jpeg_quality = jpeg_quality

    def __call__(self, image, target):
        if self.jpeg_quality is not None:
            img_buffer = BytesIO()
            image.save(img_buffer, "JPEG", quality=self.jpeg_quality)
            image = Image.open(img_buffer)

        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
        return image, target


@register_transform_class
class CustomRandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


@register_transform_class
class CustomLetterBoxResize(LetterBox):
    """
    This is really lazy/hacky. Generalize for albumentation if I require on more aggressie augmentation
    """

    def __init__(
        self,
        new_shape=(640, 640),
        auto=False,
        scaleFill=False,
        scaleup=True,
        center=True,
        stride=32,
    ):
        super().__init__(
            new_shape=new_shape,
            auto=auto,
            scaleFill=scaleFill,
            scaleup=scaleup,
            center=center,
            stride=stride,
        )

    def __call__(self, image, *args, **kwargs):
        labels = {}
        img = labels.get("img") if image is None else image
        img = np.array(image)
        # 3. Convert RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            self.new_shape[1] - new_unpad[0],
            self.new_shape[0] - new_unpad[1],
        )  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])
            ratio = (
                self.new_shape[1] / shape[1],
                self.new_shape[0] / shape[0],
            )  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 2. Convert the NumPy array to a PIL Image
        img = Image.fromarray(img)
        return img


@register_transform_class
class CustomRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


@register_transform_class
class CustomCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


# @register_transform_class
# class WeightWithKeypoints:
#     def __init__(
#         self,
#         num_features: int = 8192,
#         use_blur: bool = False,
#         clamp_min=0.1,
#     ):
#         self.detector = KF.KeyNet(pretrained=True)
#         freeze_module_params(self.detector)
#         self.num_features = num_features
#         self.use_blur = use_blur
#         self.clamp_min = clamp_min
#
#     def __call__(self, image, *args, **kwargs):
#         image = image.unsqueeze(dim=0)
#         keynet_kps = top_n_to_one(
#             self.detector(K.color.rgb_to_grayscale(image)),
#             n=self.num_features,
#             clamp_min=self.clamp_min,
#         )
#         if self.use_blur:
#             keynet_kps = K.filters.box_blur(keynet_kps, kernel_size=(5, 5))
#         return (image * keynet_kps).squeeze(dim=0)


@register_transform_class
class CustomToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


@register_transform_class
class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


@register_transform_class
class WrappedRandomResizedCrop(RandomResizedCrop):
    def __init__(self, interpolation=None, **kwargs):
        if isinstance(interpolation, str):
            interpolation = INTERPOLATION_MODE_DICT.get(interpolation, None)
        super().__init__(**kwargs, interpolation=interpolation)


@register_transform_class
class WrappedResize(Resize):
    def __init__(self, interpolation=None, **kwargs):
        if isinstance(interpolation, str):
            interpolation = INTERPOLATION_MODE_DICT.get(interpolation, None)
        super().__init__(**kwargs, interpolation=interpolation)


def get_transform(obj_name, *args, **kwargs):
    if obj_name not in TRANSFORM_CLASS_DICT:
        logger.info("No transform called `{}` is registered.".format(obj_name))
        return None
    return TRANSFORM_CLASS_DICT[obj_name](*args, **kwargs)
