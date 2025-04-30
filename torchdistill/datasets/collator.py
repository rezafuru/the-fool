from collections import abc
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from PIL.Image import Image
import torch

COLLATE_FUNC_DICT = dict()


def register_collate_func(func):
    key = (
        func.__name__
        if isinstance(func, (BuiltinMethodType, BuiltinFunctionType, FunctionType))
        else type(func).__name__
    )
    COLLATE_FUNC_DICT[key] = func
    return func


@register_collate_func
def coco_collate_fn(batch):
    return tuple(zip(*batch))


def cat_list(images, fill_value=0):
    if len(images) == 1 and not isinstance(images[0], torch.Tensor):
        return images

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


@register_collate_func
def coco_seg_collate_fn(batch):
    images, targets, supp_dicts = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets, supp_dicts


@register_collate_func
def coco_seg_eval_collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def get_collate_func(func_name):
    if func_name not in COLLATE_FUNC_DICT:
        return None
    return COLLATE_FUNC_DICT[func_name]



@register_collate_func
def default_collate_w_pillow(batch):
    r"""Puts each data field into a tensor or PIL Image with outer dimension batch size"""
    # Extended `default_collate` function in PyTorch

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate_w_pillow([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, abc.Mapping):
        return {key: default_collate_w_pillow([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_w_pillow(samples) for samples in zip(*batch)))
    elif isinstance(elem, abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_w_pillow(samples) for samples in transposed]
    elif isinstance(elem, Image):
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))
