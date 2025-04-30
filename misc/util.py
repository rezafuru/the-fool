import collections
import importlib
import os
import shutil
import uuid
from itertools import repeat

import torch
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pytorch_msssim import ssim
from torch import Tensor, nn
from PIL import Image, ImageDraw, ImageFont

from torchdistill.common import module_util
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def top_n_to_one(tensor, n, clamp_min: float = 0.0):
    """
    Set the top n numbers in the tensor to 1, and all others to 0.

    Args:
    tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
    n (int): The number of largest values to be set to 1 in each sample in the batch

    Returns:
    torch.Tensor: Tensor with the top n values set to 1 and all others to 0 for each sample in the batch
    """
    # Make sure the tensor has the right number of dimensions
    if n == -1:
        return tensor
    # Flatten spatial dimensions and sort in descending order
    values, indices = torch.flatten(tensor, start_dim=1).sort(dim=1, descending=True)

    # Get the indices of the top n values for each item in the batch
    top_indices = indices[:, :n]

    # Create a mask tensor of the same shape with all zeros
    mask = torch.zeros_like(tensor, dtype=torch.bool).flatten(start_dim=1)

    # Convert the batch indices for gather function
    batch_indices = torch.arange(tensor.size(0)).unsqueeze(-1).expand_as(top_indices)

    # Set the top values to True using scatter
    mask[batch_indices, top_indices] = True

    # Reshape the mask to original tensor shape and return it
    mask = mask.view_as(tensor)

    return torch.clamp(mask.float(), min=clamp_min)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor


def keypoints_to_spatial_tensor_with_response(
    lafs: torch.Tensor, responses: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """
    Convert keypoints to a spatial tensor and integrate responses.

    Args:
        lafs (torch.Tensor): Local Affine Frames of shape (B, N, 2, 3).
        responses (torch.Tensor): Keypoint responses of shape (B, N).
        h (int): Height of the original image.
        w (int): Width of the original image.

    Returns:
        torch.Tensor: Tensor of shape (B, 1, H, W) with keypoints marked with their responses.
    """

    # Extract x, y coordinates of keypoints from the last column of the lafs tensor
    keypoints = lafs[:, :, :, 2].round().long()
    B, N, _ = keypoints.shape

    # Create an empty tensor of the same spatial size as the image
    spatial_tensor = torch.zeros(B, 1, h, w, device=lafs.device, dtype=torch.float32)

    # Construct indexing arrays for batch and keypoints
    b_idx = torch.arange(B, device=lafs.device).view(B, 1).expand(-1, N).reshape(-1)
    n_idx = torch.arange(N, device=lafs.device).view(1, N).expand(B, -1).reshape(-1)

    # Ensure the keypoints are within the image boundaries
    x_coords = keypoints[:, :, 0].clamp(0, w - 1).reshape(-1)
    y_coords = keypoints[:, :, 1].clamp(0, h - 1).reshape(-1)

    # Mark keypoints on the spatial tensor with their responses
    spatial_tensor[b_idx, 0, y_coords, x_coords] = responses.reshape(-1)

    return spatial_tensor


def highest_similarity_iter(images):
    """
    Computes the image tensor which has the highest average SSIM with all other image tensors.

    Parameters:
    - images (List[torch.Tensor]): A list of image tensors of shape (1, C, H, W).

    Returns:
    - torch.Tensor: Image tensor of shape (1, C, H, W) with highest average similarity.
    """

    num_images = len(images)
    ssim_scores = []

    for idx, image in enumerate(images):
        # Repeat the image tensor to match the number of images for batch processing.
        X1 = image.repeat(num_images, 1, 1, 1)
        X2 = torch.cat(images, 0)  # Stack all image tensors.

        # Compute SSIM scores.
        scores = ssim(X1, X2, data_range=255, size_average=False)

        # Exclude the SSIM score of the image with itself and take the average.
        avg_score = (torch.sum(scores) - scores[idx]) / (num_images - 1)
        ssim_scores.append(avg_score)

    # Convert list to tensor for efficient operations.
    ssim_scores_tensor = torch.tensor(ssim_scores)

    # Find the index of the image with the highest average similarity.
    max_idx = torch.argmax(ssim_scores_tensor)

    return max_idx


def separate_keyframes_and_stack_frames(highest_indices: Tensor, frames: List[Tensor]):
    """
    Constructs two tensors based on the highest similarity indices and the original images.

    Parameters:
    - highest_indices (torch.Tensor): A tensor of shape (B,) containing the index of the tensor with the highest similarity for each batch.
    - frames (List[torch.Tensor]): A list of image tensors of shape (B, C, H, W).

    Returns:
    - A tuple containing two tensors:
        1. A tensor of shape (B, D, C, H, W) where D is the number of repeated keyframe.
        2. A tensor of shape (B, C, H, W) stacking the tensors that are not the keyframe .
    """

    B, C, H, W = frames[0].shape
    D = len(frames)

    # Stack all images into a single tensor of shape (B, D, C, H, W)
    all_frames = torch.stack(frames, dim=1)

    # Create a mask that is True where the index is not the highest
    mask = torch.arange(D, device=all_frames.device).view(1, D).expand(
        B, D
    ) != highest_indices.view(B, 1)

    # Expand the mask to match the dimensions of all_images
    expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, D, C, H, W)

    # Use the mask to gather the non-highest images
    other_frames = all_frames[expanded_mask].view(B, D - 1, C, H, W)

    # Use the highest_indices to gather the highest images and repeat them along the D dimension
    keyframe = all_frames[torch.arange(B), highest_indices]

    return keyframe, other_frames


def highest_similarity_batch(frames):
    """
    Computes the image tensor which has the highest average SSIM with all other image tensors for each batch.

    Parameters:
    - images (List[torch.Tensor]): A list of image tensors of shape (B, C, H, W).

    Returns:
    - A tensor of shape (B,) containing the index of the tensor with the highest similarity for each batch.
    """

    num_images = len(frames)
    B, C, H, W = frames[0].shape

    # Construct the Cartesian product for the image combinations
    X1 = torch.cat([img.repeat(1, num_images, 1, 1, 1) for img in frames], 1).view(
        -1, C, H, W
    )
    X2 = torch.cat(frames, 1).repeat(1, num_images, 1, 1, 1).view(-1, C, H, W)

    # Compute SSIM scores for all combinations.
    scores = ssim(
        X1, X2, data_range=1.0, size_average=False
    )  # Shape: (B * num_images * num_images)

    # Reshape the scores tensor to separate results per image.
    scores_matrix = scores.view(B, num_images, num_images)

    # Compute average SSIM per image excluding self comparisons.
    diag_indices = torch.arange(0, num_images)
    scores_matrix[:, diag_indices, diag_indices] = 0  # Setting diagonal values to 0
    avg_scores = scores_matrix.sum(2) / (num_images - 1)  # Shape: (B, num_images)

    # Find the index of the image with the highest average similarity for each batch.
    max_idx = torch.argmax(avg_scores, dim=1)  # Shape: (B,)

    return max_idx


def highest_similarity(images):
    """
    Computes the image tensor which has the highest average SSIM with all other image tensors.

    Parameters:
    - images (List[torch.Tensor]): A list of image tensors of shape (1, C, H, W).

    Returns:
    - torch.Tensor: Image tensor of shape (1, C, H, W) with highest average similarity.
    """

    num_images = len(images)

    # Construct the Cartesian product for the image combinations
    X1 = torch.cat([img.repeat(num_images, 1, 1, 1) for img in images], 0)
    X2 = torch.cat(images * num_images, 0)

    # Compute SSIM scores for all combinations.
    scores = ssim(X1, X2, data_range=255, size_average=False)

    # Reshape the scores tensor to separate results per image.
    scores_matrix = scores.view(num_images, num_images)

    # Compute average SSIM per image excluding self comparisons.
    diag_indices = torch.arange(0, num_images)
    scores_matrix[diag_indices, diag_indices] = 0  # Setting diagonal values to 0
    avg_scores = scores_matrix.sum(1) / (num_images - 1)

    # Find the index of the image with the highest average similarity.
    max_idx = torch.argmax(avg_scores)

    return max_idx


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_iou_types(model) -> List[str]:
    iou_type_list = ["bbox"]
    # note: only if I ever bother with segmentation and keypoint
    # if isinstance(model, MaskRCNN):
    #     iou_type_list.append('segm')
    # if isinstance(model, KeypointRCNN):
    #     iou_type_list.append('keypoints')
    return iou_type_list


def get_module(module_path):
    """
    Return a module reference
    """
    module_ = importlib.import_module(module_path)
    return module_


def short_uid() -> str:
    return str(uuid.uuid4())[0:8]


def check_if_module_exits(module, module_path) -> bool:
    module_names = module_path.split(".")
    child_module_name = module_names[0]
    if len(module_names) == 1:
        return hasattr(module, child_module_name)

    if not hasattr(module, child_module_name):
        return False
    return check_if_module_exits(
        getattr(module, child_module_name), ".".join(module_names[1:])
    )


def extract_entropy_bottleneck_module(model) -> nn.Module:
    model_wo_ddp = model.module if module_util.check_if_wrapped(model) else model
    entropy_bottleneck_module = None
    if check_if_module_exits(model_wo_ddp, "compression_module.entropy_bottleneck"):
        entropy_bottleneck_module = module_util.get_module(
            model_wo_ddp, "compression_module"
        )
    elif check_if_module_exits(model_wo_ddp, "compression_model.entropy_bottleneck"):
        entropy_bottleneck_module = module_util.get_module(
            model_wo_ddp, "compression_model"
        )
    return entropy_bottleneck_module


def chmod_r(path: str, mode: int):
    """Recursive chmod"""
    if not os.path.exists(path):
        return
    os.chmod(path, mode)
    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            os.chmod(os.path.join(root, dirname), mode)
        for filename in filenames:
            os.chmod(os.path.join(root, filename), mode)


def rm_rf(path: str):
    """
    Recursively removes a file or directory
    """
    if not path or not os.path.exists(path):
        return
    try:
        chmod_r(path, 0o777)
    except PermissionError:
        pass
    exists_but_non_dir = os.path.exists(path) and not os.path.isdir(path)
    if os.path.isfile(path) or exists_but_non_dir:
        os.remove(path)
    else:
        shutil.rmtree(path)


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def uniquify(path) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


class Tokenizer(nn.Module):
    """
    Patch embed without Projection (From Image Tensor to Token Tensor)
    """

    def __init__(self):
        super(Tokenizer, self).__init__()

    def forward(self, x) -> Tensor:
        x = x.flatten(2).transpose(1, 2)  # B h*w C
        return x


class Detokenizer(nn.Module):
    """
    Inverse operation of Tokenizer (From Token Tensor to Image Tensor)
    """

    def __init__(self, spatial_dims):
        super(Detokenizer, self).__init__()
        self.spatial_dims = spatial_dims

    def forward(self, x) -> Tensor:
        B, _, C = x.shape
        H, W = self.spatial_dims
        return x.transpose(1, 2).view(B, -1, H, W)


def overwrite_config(org_config, sub_config):
    def _isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    for sub_key, sub_value in sub_config.items():
        if sub_key in org_config:
            if isinstance(sub_value, dict):
                overwrite_config(org_config[sub_key], sub_value)
            else:
                org_config[sub_key] = (
                    float(sub_value) if _isfloat(sub_value) else sub_value
                )
        else:
            org_config[sub_key] = sub_value


def show_att_map_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    map = (1 - image_weight) * heatmap + image_weight * img
    map = map / np.max(map)
    return np.uint8(255 * map)


def concat_images_h(
    img_a: Image.Image, img_b: Image.Image, margin: Optional[int] = None
) -> Image.Image:
    dst = Image.new("RGB", (img_b.width + img_a.width, img_b.height))
    dst.paste(img_b, (0, 0))
    dst.paste(img_a, (img_b.width, 0))
    return dst


def concat_images_h_caption_metric(
    img_a: Image.Image,
    img_b: Tuple[Image.Image, Optional[str]],
    metric: Optional[str] = None,
) -> Image.Image:
    img_b, metric_val = img_b
    if metric:
        recon_img_cap = ImageDraw.Draw(img_b)
        recon_img_cap.text(
            (0, 0),
            f"{metric}={metric_val}",
            font=ImageFont.truetype("FreeMono.ttf", 30),
            fill=(0, 0, 0),
        )
    dst = Image.new("RGB", (img_b.width + img_a.width, img_b.height))
    dst.paste(img_b, (5, 5))
    dst.paste(img_a, (img_b.width, 0))
    return dst


def concat_images_v(
    img_a: Image.Image, img_b: Image.Image, margin: Optional[int] = None
) -> Image.Image:
    dst = Image.new("RGB", (img_b.width, img_b.height + img_a.height))
    dst.paste(img_b, (0, 0))
    dst.paste(img_a, (0, img_b.height))
    return dst


def concat_images_v2(frames: List[Image.Image]) -> Image.Image:
    ws, hs = zip(*(i.size for i in frames))

    dst = Image.new("RGB", (max(ws), sum(hs)))

    y_offset = 0
    for frame in frames:
        dst.paste(frame, (0, y_offset))
        y_offset += frame.height

    return dst


def concat_frames_v(
    frames_a: List[Image.Image],
    frames_b: List[Image.Image],
):
    assert len(frames_a) == len(frames_b)
    dst = Image.new(
        "RGB",
        (
            max(frames_a[0].width * len(frames_a), frames_b[0].width * len(frames_b)),
            frames_a[0].height + frames_b[0].height,
        ),
    )
    for i, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b)):
        dst.paste(frame_b, (frame_b.width * i, 0))
        dst.paste(frame_a, (frame_a.width * i, frame_b.height))
    return dst


def concat_frames_h(
    frames_a: List[Image.Image], frames_b: List[Image.Image]
) -> Image.Image:
    assert len(frames_a) == len(frames_b)
    dst = Image.new(
        "RGB",
        (
            frames_a[0].width + frames_b[0].width,
            max(frames_a[0].height * len(frames_a), frames_b[0].height * len(frames_b)),
        ),
    )
    for i, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b)):
        dst.paste(frame_b, (0, frame_b.height * i))
        dst.paste(frame_a, (frame_b.width, frame_a.height * i))
    return dst


def concat_images_h(frames: List[Image.Image], margin: int = 0) -> Image.Image:
    ws, hs = zip(*(i.size for i in frames))

    total_width = sum(ws) + margin * (len(frames) - 1)
    max_height = max(hs)

    dst = Image.new(
        "RGB",
        (total_width, max_height),
        color="white",  # Set the default color to white
    )

    x_offset = 0
    for frame in frames:
        dst.paste(frame, (x_offset, 0))
        x_offset += frame.width + margin  # Add the margin after each image

    return dst


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_groups, channels)
