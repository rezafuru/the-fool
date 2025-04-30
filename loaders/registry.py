import copy
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from compressai.datasets import VideoFolder, Vimeo90kDataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from torchdistill.common.constant import def_logger
from torchdistill.common.module_util import freeze_module_params
from torchdistill.datasets.registry import register_dataset
from torchdistill.models.registry import get_model

logger = def_logger.getChild(__name__)


@register_dataset
class Vimeo90kDataset(Dataset):
    """
    Original from https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/vimeo90k.py

    Load a Vimeo-90K structured dataset.

    Vimeo-90K dataset from
    Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, William T. Freeman:
    `"Video Enhancement with Task-Oriented Flow"
    <https://arxiv.org/abs/1711.09078>`_,
    International Journal of Computer Vision (IJCV), 2019.

    Training and testing image samples are respectively stored in
    separate directories:

    .. code-block::

        - rootdir/
            - sequence/
                - 00001/001/im1.png
                - 00001/001/im2.png
                - 00001/001/im3.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'valid')
        tuplet (int): order of dataset tuplet (e.g. 3 for "triplet" dataset)
    """

    def __init__(self, root, transform=None, split="train", tuplet=3, *args, **kwargs):
        root = os.path.expanduser(root)
        list_path = Path(root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            # self.samples = [
            #     f"{root}/sequences/{line.rstrip()}/im{idx}.png"
            #     for line in f
            #     if line.strip() != ""
            #     for idx in range(1, tuplet + 1)
            # ]
            self.samples = [
                f_sub
                for line in f
                if (Path(root) / "sequences" / line.strip()).is_dir()
                for f_sub in (Path(root) / "sequences" / line.strip()).iterdir()
                if f_sub.is_file()
            ]

        self.transform = transform or nn.Identity()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = self.transform(Image.open(self.samples[index]).convert("RGB"))
        return img, img.detach().clone()  # todo: ask matsubara about a better way

    def __len__(self):
        return len(self.samples)

    def _list_filename(self, split: str, tuplet: Optional[int] = None) -> str:
        # if tuplet:
        #     tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        #     return f"{tuplet_prefix}_{split}.list"
        return f"{split}.list"


@register_dataset
class VideoFolder(Dataset):
    """


    Original from https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/datasets/video.py
       Args:
       root (string): root directory of the dataset
       rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
       transform (callable, optional): a function or transform that takes in a
           PIL image and returns a transformed version
       split (string): split mode ('train' or 'test')
    """

    def __init__(
            self,
            root: str,
            rnd_interval: bool = False,
            rnd_temp_order: bool = False,
            transform: Callable = None,
            split: str = "train",
            max_frames: int = 3,
            *args,
            **kwargs,
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")
        root = os.path.expanduser(root)
        splitfile = Path(f"{root}/{split}.list")

        if splitfile.exists():
            splitdir = Path(f"{root}/sequences")
            assert splitfile.is_file(), f"Invalid file '{splitfile}'"
            assert splitdir.is_dir(), f"Invalid directory '{splitdir}'"
            with open(splitfile, "r") as f_in:
                self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]
        else:
            logger.info(
                "Treating Videofolder instance as a sequenced detection dataset.."
            )
            splitdir = Path(root) / "images" / split
            # self.sample_folders = [subdir for subdir in splitdir.iterdir() if subdir.is_dir()]
            # self.sample_folders = [subdir for subdir in splitdir.glob("**/*") if subdir.is_dir()]
            self.sample_folders = [
                subdir for subdir in splitdir.rglob("*/*") if subdir.is_dir()
            ]
            # if not list(splitdir.iterdir()) == self.sample_folders:
            #     logger.warning(f"{splitdir} has non sequence files (that probably shouldn't be there)")

        self.max_frames = max_frames
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]
        # print(frame_paths)
        frames = self.transform(
            np.concatenate(
                [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
            )
        )
        if self.rnd_temp_order:
            frames = list(torch.chunk(frames, self.max_frames))
            random.shuffle(frames)
            frames_target = [f.detach().clone() for f in frames]
            return tuple(frames), tuple(frames_target)
        frames_target = frames.detach().clone()
        frames = torch.chunk(frames, self.max_frames)
        frames_target = torch.chunk(frames_target, self.max_frames)

        return frames, frames_target

    def __len__(self):
        return len(self.sample_folders)


@register_dataset
class MultiVideoFolder(VideoFolder):
    """ """

    def __init__(
            self,
            roots: List[str],
            rnd_interval: bool = False,
            rnd_temp_order: bool = False,
            transform: Callable = None,
            split: str = "train",
            max_frames: int = 3,
            *args,
            **kwargs,
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")
        roots = [os.path.expanduser(root) for root in roots]

        splitdirs = [Path(root) / "images" / split for root in roots]
        # self.sample_folders = [subdir for subdir in splitdir.iterdir() if subdir.is_dir()]
        # self.sample_folders = [subdir for subdir in splitdir.glob("**/*") if subdir.is_dir()]

        self.sample_folders = [
            subdir
            for splitdir in splitdirs
            for subdir in splitdir.rglob("*/*")
            if subdir.is_dir()
        ]
        # if not list(splitdir.iterdir()) == self.sample_folders:
        #     logger.warning(f"{splitdir} has non sequence files (that probably shouldn't be there)")

        self.max_frames = max_frames
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform


@register_dataset
class MultiVideoFeatureFolder(MultiVideoFolder):
    """ """

    def __init__(
            self,
            roots: List[str],
            feature_extractor_config: dict,
            rnd_interval: bool = False,
            rnd_temp_order: bool = False,
            transform: Callable = None,
            split: str = "train",
            max_frames: int = 3,
            *args,
            **kwargs,
    ):
        super().__init__(
            roots, rnd_interval, rnd_temp_order, transform, split, max_frames
        )

        self.feature_extractor = get_model(
            feature_extractor_config["name"], **feature_extractor_config["params"]
        )
        freeze_module_params(self.feature_extractor)
        self.feature_extractor.eval()
        self.feature_extractor.train = lambda: None

    def __getitem__(self, index: int):
        sample_folder = self.sample_folders[index]
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]
        # print(frame_paths)
        frames = self.transform(
            np.concatenate(
                [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
            )
        )
        if self.rnd_temp_order:
            frames = list(torch.chunk(frames, self.max_frames))
            random.shuffle(frames)
            frames_target = [f.detach().clone() for f in frames]
            return tuple(frames), tuple(frames_target)
        frames_target = frames.detach().clone()
        frames_features = self.feature_extractor([t.unsqueeze(dim=0) for t in frames.chunk(self.max_frames)])
        frames_features = torch.chunk(frames_features, self.max_frames)
        frames_target = torch.chunk(frames_target, self.max_frames)

        return frames_features, frames_target


@register_dataset
class ImageFolderNoLabel(Dataset):

    def __init__(self, root, transform=None, target_transform=None, split=None):
        root = os.path.expanduser(root)
        if split:
            root = Path(root) / split
        else:
            root = Path(root)

        # store paths to all files in the directory
        self.samples = [str(f) for f in root.iterdir() if f.is_file()]

        self.transform = transform or nn.Identity()
        self.target_transform = target_transform or nn.Identity()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = self.transform(Image.open(self.samples[index]).convert("RGB"))
        return img, img.detach().clone()

    def __len__(self):
        return len(self.samples)


@register_dataset
class MultiImageFolderNoLabel(ImageFolderNoLabel):
    """

    Each root should have the required folder strucutre of ImageFolderNoLabel and have the same split subdir
    """

    def __init__(self, roots, transform=None, target_transform=None, split=None):
        roots = [os.path.expanduser(root) for root in roots]
        if split:
            roots = [Path(root) / split for root in roots]
        else:
            roots = [Path(root) for root in roots]

        # store paths to all files in the directory
        self.samples = []
        for root in roots:
            sublist = []
            for f in root.iterdir():
                if f.is_file():
                    sublist.append(str(f))
            assert len(sublist) > 0
            self.samples.extend(sublist)

        # self.samples = [str(f) for root in roots for f in root.iterdir() if f.is_file()]

        self.transform = transform or nn.Identity()
        self.target_transform = target_transform or nn.Identity()


@register_dataset
class ImageFolderFilenameAsLabel(Dataset):
    def __init__(self, root, transform=None, target_transform=None, split=None):
        root = os.path.expanduser(root)
        if split:
            root = Path(root) / split
        else:
            root = Path(root)

        # store paths to all files in the directory
        self.samples = [str(f) for f in root.iterdir() if f.is_file()]

        self.transform = transform or nn.Identity()
        self.target_transform = target_transform or nn.Identity()

    def __getitem__(self, index):
        img_orig = Image.open(self.samples[index]).convert("RGB")
        img = self.transform(copy.deepcopy(img_orig))
        return img, Path(self.samples[index]).name, img_orig

    def __len__(self):
        return len(self.samples)


torchvision.datasets.__dict__["Vimeo90kDataset"] = Vimeo90kDataset
torchvision.datasets.__dict__["VideoFolder"] = VideoFolder
torchvision.datasets.__dict__["ImageFolderNoLabel"] = ImageFolderNoLabel
torchvision.datasets.__dict__["MultiImageFolderNoLabel"] = MultiImageFolderNoLabel
torchvision.datasets.__dict__["MultiVideoFolder"] = MultiVideoFolder
torchvision.datasets.__dict__["MultiVideoFeatureFolder"] = MultiVideoFeatureFolder
torchvision.datasets.__dict__["ImageFolderFilenameAsLabel"] = ImageFolderFilenameAsLabel
