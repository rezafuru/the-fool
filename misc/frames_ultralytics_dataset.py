import glob
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.data.dataset import (
    DATASET_CACHE_VERSION,
    load_dataset_cache_file,
)
from ultralytics.data.utils import (
    HELP_URL,
    IMG_FORMATS,
    get_hash,
    verify_image_label,
)
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    NUM_THREADS,
    TQDM,
    colorstr,
    is_dir_writeable,
)
from ultralytics.utils.instance import Instances

from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)

"""
Reminder: need to recreate logic of creating sequenced dataset, to not tile the dataset but use a tiled dataset already
 + add labels and augmented samples
"""

def build_sequenced_yolo_dataset(
    cfg, img_path, batch, data, mode="val", rect=False, stride=32
):
    """Build YOLO Dataset"""
    return SequencedYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        use_segments=cfg.task == "segment",
        use_keypoints=cfg.task == "pose",
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        split=mode
    )

def build_frames_yolo_dataset(
    cfg, img_path, batch, data, mode="val", rect=False, stride=32
):
    """Build YOLO Dataset"""
    return FramesYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        use_segments=cfg.task == "segment",
        use_keypoints=cfg.task == "pose",
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        Path(f"{path}.npy").rename(path)  # remove .npy suffix
        logger.info(f"{prefix}New cache created: {path}")
    else:
        logger.warning(
            f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved."
        )


def img2label_paths(img_paths) -> List[str]:
    """Define label paths as a function of image paths."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def sequenced_grouped_img2label_paths(img_paths) -> Dict[str, List[str]]:
    """Define label paths as a function of image paths."""
    res = defaultdict(list)
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )
    for prefix, prefix_img_paths in img_paths.items():
        # /images/, /labels/ substrings
        res[prefix] = [
            sb.join(x.__str__().rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
            for x in prefix_img_paths
        ]
    return res


def grouped_img2label_paths(img_paths) -> Dict[str, List[str]]:
    """Define label paths as a function of image paths."""
    res = defaultdict(list)
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )
    for prefix, prefix_img_paths in img_paths.items():
        # /images/, /labels/ substrings
        res[prefix] = [
            sb.join(x.__str__().rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
            for x in prefix_img_paths
        ]
    return res


class SequencedFramesUltralyticsDataset(Dataset):
    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=False,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        split="val",
            sequence_length=5
    ):
        super().__init__()
        self.root_path = Path(img_path).parent if ".txt" in img_path else Path(img_path)
        self.split = split
        assert not classes, "implement if necessary"
        self.use_keypoints = False
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.sequence_length = sequence_length
        # todo: consider sequence length
        self.image_groups, self.prefix_list = self.get_sequenced_image_groups()
        self.grouped_label_files = sequenced_grouped_img2label_paths(self.image_groups)
        self.grouped_labels = self.get_grouped_labels()
        # doesn't seem to be necessary for my use case
        # self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = {
            prefix: len(group) for prefix, group in self.image_groups.items()
        }  # number of images per group
        assert not self.augment
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        # self.max_buffer_length = (
        #     min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        # )

        # Cache stuff
        if cache == "ram" and not self.check_cache_ram():
            cache = False

        self.ims, self.im_hw0, self.im_hw = {}, {}, {}
        for prefix in self.prefix_list:
            self.ims[prefix], self.im_hw0[prefix], self.im_hw[prefix] = (
                [None] * len(self.image_groups[prefix]),
                [None] * len(self.image_groups[prefix]),
                [None] * len(self.image_groups[prefix]),
            )
        self.npy_files = {
            prefix: [Path(f).with_suffix(".npy") for f in self.image_groups[prefix]]
            for prefix in self.prefix_list
        }

        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def create_sample_groups(self):
        self.image_groups = defaultdict(list)
        self.label_groups = defaultdict(list)

        image_dir = Path(self.root_path) / "images" / "val"
        label_dir = Path(self.root_path) / "labels" / "val"

        for image_path in sorted(image_dir.glob("*.png")):
            prefix = image_path.stem.split("_")[0]
            self.image_groups[prefix].append(image_path)

        for label_path in sorted(label_dir.glob("*.txt")):
            prefix = label_path.stem.split("_")[0]
            self.label_groups[prefix].append(label_path)

        self.prefix_list = list(self.image_groups.keys())

    def get_sequenced_image_groups(self):
        """
        root/sequences/{prefix}/{prefix_group}

        One group per prefix group. After init, we don't need to care about sequence subdir only prefix_group subdir
        """
        assert Path(self.root_path).is_dir(), "Don't use .txt file for this"
        # In a dir with subdirs, get all subdirs of each subdir

        prefix_group_dirs = glob.glob(f"{self.root_path}/images/{self.split}/*/*/")
        image_groups = defaultdict(list)
        for prefix_group_dir in prefix_group_dirs:
            for image_path in sorted(Path(prefix_group_dir).glob("*.png")):
                sequence = image_path.stem.split("_")[0]
                group = image_path.parent.name
                prefix = f"{sequence}/{group}"
                image_groups[prefix].append(image_path)
        prefix_list = list(image_groups.keys())
        return image_groups, prefix_list

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x
                            for x in t
                        ]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(
                x.replace("/", os.sep)
                for x in f
                if x.split(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [
                        segments[si] for si, idx in enumerate(j) if idx
                    ]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, prefix, group_index, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = (
            self.ims[prefix][group_index],
            self.image_groups[prefix][group_index],
            self.npy_files[prefix][group_index],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f.__str__())  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (
                h0 == w0 == self.imgsz
            ):  # resize by stretching image to square imgsz
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

            # Add to buffer if training with augmentations
            # if self.augment:
            #     self.ims[prefix][group_index], self.im_hw0[prefix][group_index], self.im_hw[prefix][group_index] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
            #     self.buffer.append(group_index)
            #     if len(self.buffer) >= self.max_buffer_length:
            #         j = self.buffer.pop(0)
            #         self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return (
            self.ims[prefix][group_index],
            self.im_hw0[prefix][group_index],
            self.im_hw[prefix][group_index],
        )

    def load_image_group(self, group_index, rect_mode=True):
        res = []
        prefix = self.prefix_list[group_index]
        for i in range(len(self.image_groups[prefix])):
            im, f, fn = (
                self.ims[prefix][i],
                self.image_groups[prefix][i],
                self.npy_files[prefix][i],
            )
            if im is None:  # not cached in RAM
                if fn.exists():  # load npy
                    im = np.load(fn)
                else:  # read image
                    im = cv2.imread(f.__str__())  # BGR
                    if im is None:
                        raise FileNotFoundError(f"Image Not Found {f}")
                h0, w0 = im.shape[:2]  # orig hw
                if (
                    rect_mode
                ):  # resize long side to imgsz while maintaining aspect ratio
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (
                            min(math.ceil(w0 * r), self.imgsz),
                            min(math.ceil(h0 * r), self.imgsz),
                        )
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                elif not (
                    h0 == w0 == self.imgsz
                ):  # resize by stretching image to square imgsz
                    im = cv2.resize(
                        im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                    )

                # Add to buffer if training with augmentations
                # if self.augment:
                #     self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                #         im,
                #         (h0, w0),
                #         im.shape[:2],
                #     )  # im, hw_original, hw_resized
                #     self.buffer.append(i)
                #     if len(self.buffer) >= self.max_buffer_length:
                #         j = self.buffer.pop(0)
                #         self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

                res.append((im, (h0, w0), im.shape[:2]))
            else:
                res.append(
                    (self.ims[prefix][i], self.im_hw0[prefix][i], self.im_hw[prefix][i])
                )
        return res

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image_group
        for prefix, prefix_img_paths in self.image_groups.items():
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(self.ni))
                pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, x in pbar:
                    if cache == "disk":
                        b += self.npy_files[i].stat().st_size
                    else:  # 'ram'
                        (
                            self.ims[i],
                            self.im_hw0[i],
                            self.im_hw[i],
                        ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                        b += self.ims[i].nbytes
                    pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {cache})"
                pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(sum(self.ni.values()), 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = (
            b * sum(self.ni.values()) / n * (1 + safety_margin)
        )  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = (
            mem_required < mem.available
        )  # to cache or not to cache, that is the question
        if not cache:
            logger.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int)
            * self.stride
        )
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Should return all samples from exactly one sequence/group per call"""
        images_and_labels = self.get_grouped_image_and_label(index)
        return [
            self.transforms(image_and_label) for image_and_label in images_and_labels
        ]

    def get_grouped_image_and_label(self, index) -> List[Dict[str, any]]:
        """
        Extend to return a list of dictionaries instead of a single dictionary
        """
        """Get and return label information from the dataset."""
        prefix = self.prefix_list[index]
        labels = deepcopy(
            self.grouped_labels[prefix]
        )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        res = []
        for group_index, label in enumerate(labels):
            label.pop("shape", None)  # shape is for rect, remove it
            (
                label["img"],
                label["ori_shape"],
                label["resized_shape"],
            ) = self.load_image(prefix=prefix, group_index=group_index)
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )  # for evaluation
            if self.rect:
                label["rect_shape"] = self.batch_shapes[self.batch[group_index]]
            res.append(self.update_labels_info(label))
        return res

    def get_image_and_label(self, index) -> Dict[str, any]:
        """Get and return label information from the dataset."""
        label = deepcopy(
            self.labels[index]
        )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        (
            label["img"],
            label["ori_shape"],
            label["resized_shape"],
        ) = self.load_image_group(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.grouped_labels)

    def update_labels_info(self, label):
        """custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_grouped_labels(self):
        # should be 15148 labels in total

        cache_path = Path(self.root_path / "labels" / self.split / "cache_grouped")
        # cache_path = Path(
        #     list(self.grouped_label_files.values())[0][0]
        # ).parent.with_suffix(".cache_grouped")
        try:
            cache, exists = (
                # shoul still work as before, instead of loading a dict we load a list of dicts
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.grouped_label_files) + get_hash(
                self.image_groups
            )
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = (
                self.cache_labels_grouped(cache_path),
                False,
            )  # run cache ops

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        grouped_labels = cache["grouped_labels"]
        # orig is a dict with a single key (labels) with a list of dicts (one element per sample)
        if not grouped_labels:
            logger.warning(
                f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        # self.im_files = [lb["im_file"] for lb in grouped_labels]  # update im_fi
        self.image_groups = {
            prefix: [sample["im_file"] for sample in gr]
            for prefix, gr in grouped_labels.items()
        }

        return grouped_labels

    def cache_labels_grouped(self, path=Path("./labels.cache_gouped")):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        res = {}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        # total = len(self.im_files)
        # total_by_group = dict(map(lambda x: len(x), self.image_groups.values()))
        total_by_group = {k: len(v) for k, v in self.image_groups.items()}
        assert not self.use_keypoints, "Not needed for our use case"
        nkpt, ndim = 0, 0
        for prefix, total in total_by_group.items():
            x = []
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(
                    func=verify_image_label,
                    iterable=zip(
                        self.image_groups[prefix],
                        self.grouped_label_files[prefix],
                        repeat(self.prefix),
                        repeat(self.use_keypoints),
                        repeat(len(self.data["names"])),
                        repeat(nkpt),
                        repeat(ndim),
                    ),
                )
                pbar = TQDM(results, desc=desc, total=total)
                for (
                    im_file,
                    lb,
                    shape,
                    segments,
                    keypoint,
                    nm_f,
                    nf_f,
                    ne_f,
                    nc_f,
                    msg,
                ) in pbar:
                    nm += nm_f
                    nf += nf_f
                    ne += ne_f
                    nc += nc_f
                    if im_file:
                        x.append(
                            dict(
                                im_file=im_file,
                                shape=shape,
                                cls=lb[:, 0:1],  # n, 1
                                bboxes=lb[:, 1:],  # n, 4
                                segments=segments,
                                keypoints=keypoint,
                                normalized=True,
                                bbox_format="xywh",
                            )
                        )
                    if msg:
                        msgs.append(msg)
                    pbar.desc = (
                        f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                    )
                pbar.close()
            res[prefix] = x
        res = {"grouped_labels": res}
        if msgs:
            logger.info("\n".join(msgs))
        if nf == 0:
            logger.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}.")
        res["hash"] = get_hash(self.grouped_label_files) + get_hash(self.image_groups)
        res["results"] = nf, nm, ne, nc, len(self.image_groups)
        res["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, res)
        return res


class FramesUltralyticsDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=False,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        sequenced=False,
    ):
        super().__init__()
        self.root_path = Path(img_path).parent if ".txt" in img_path else Path(img_path)
        assert "train" not in self.root_path.__str__(), "Only use this for validation"
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.image_groups, self.prefix_list = self.get_grouped_img_files(
            Path(self.root_path) / "images" / "val"
            if "val" not in self.root_path.__str__()
            else Path(self.root_path)
        )
        self.grouped_labels = self.get_grouped_labels()

        # doesn't seem to be necessary for my use case
        # self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = {
            prefix: len(group) for prefix, group in self.image_groups.items()
        }  # number of images per group
        assert not self.augment
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        # self.max_buffer_length = (
        #     min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        # )

        # Cache stuff
        if cache == "ram" and not self.check_cache_ram():
            cache = False

        self.ims, self.im_hw0, self.im_hw = {}, {}, {}
        for prefix in self.prefix_list:
            self.ims[prefix], self.im_hw0[prefix], self.im_hw[prefix] = (
                [None] * len(self.image_groups[prefix]),
                [None] * len(self.image_groups[prefix]),
                [None] * len(self.image_groups[prefix]),
            )
        # self.ims, self.im_hw0, self.im_hw = (
        #     [None] * self.ni,
        #     [None] * self.ni,
        #     [None] * self.ni,
        # )
        # assert not cache, "Implement if useful"
        self.npy_files = {
            prefix: [Path(f).with_suffix(".npy") for f in self.image_groups[prefix]]
            for prefix in self.prefix_list
        }

        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def create_sample_groups(self):
        self.image_groups = defaultdict(list)
        self.label_groups = defaultdict(list)

        image_dir = Path(self.root_path) / "images" / "val"
        label_dir = Path(self.root_path) / "labels" / "val"

        for image_path in sorted(image_dir.glob("*.png")):
            prefix = image_path.stem.split("_")[0]
            self.image_groups[prefix].append(image_path)

        for label_path in sorted(label_dir.glob("*.txt")):
            prefix = label_path.stem.split("_")[0]
            self.label_groups[prefix].append(label_path)

        self.prefix_list = list(self.image_groups.keys())

    def get_grouped_img_files(self, img_path):
        assert Path(img_path).is_dir(), "Don't use .txt file for this"
        f = []
        image_groups = defaultdict(list)
        for image_path in sorted(img_path.glob("*.png")):
            prefix = image_path.stem.split("_")[0]
            image_groups[prefix].append(image_path)
        prefix_list = list(image_groups.keys())
        return image_groups, prefix_list

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x
                            for x in t
                        ]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(
                x.replace("/", os.sep)
                for x in f
                if x.split(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [
                        segments[si] for si, idx in enumerate(j) if idx
                    ]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, prefix, group_index, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = (
            self.ims[prefix][group_index],
            self.image_groups[prefix][group_index],
            self.npy_files[prefix][group_index],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f.__str__())  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (
                h0 == w0 == self.imgsz
            ):  # resize by stretching image to square imgsz
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

            # Add to buffer if training with augmentations
            # if self.augment:
            #     self.ims[prefix][group_index], self.im_hw0[prefix][group_index], self.im_hw[prefix][group_index] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
            #     self.buffer.append(group_index)
            #     if len(self.buffer) >= self.max_buffer_length:
            #         j = self.buffer.pop(0)
            #         self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return (
            self.ims[prefix][group_index],
            self.im_hw0[prefix][group_index],
            self.im_hw[prefix][group_index],
        )

    def load_image_group(self, group_index, rect_mode=True):
        res = []
        prefix = self.prefix_list[group_index]
        for i in range(len(self.image_groups[prefix])):
            im, f, fn = (
                self.ims[prefix][i],
                self.image_groups[prefix][i],
                self.npy_files[prefix][i],
            )
            if im is None:  # not cached in RAM
                if fn.exists():  # load npy
                    im = np.load(fn)
                else:  # read image
                    im = cv2.imread(f.__str__())  # BGR
                    if im is None:
                        raise FileNotFoundError(f"Image Not Found {f}")
                h0, w0 = im.shape[:2]  # orig hw
                if (
                    rect_mode
                ):  # resize long side to imgsz while maintaining aspect ratio
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (
                            min(math.ceil(w0 * r), self.imgsz),
                            min(math.ceil(h0 * r), self.imgsz),
                        )
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                elif not (
                    h0 == w0 == self.imgsz
                ):  # resize by stretching image to square imgsz
                    im = cv2.resize(
                        im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                    )

                # Add to buffer if training with augmentations
                # if self.augment:
                #     self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                #         im,
                #         (h0, w0),
                #         im.shape[:2],
                #     )  # im, hw_original, hw_resized
                #     self.buffer.append(i)
                #     if len(self.buffer) >= self.max_buffer_length:
                #         j = self.buffer.pop(0)
                #         self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

                res.append((im, (h0, w0), im.shape[:2]))
            else:
                res.append(
                    (self.ims[prefix][i], self.im_hw0[prefix][i], self.im_hw[prefix][i])
                )
        return res

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image_group
        for prefix, prefix_img_paths in self.image_groups.items():
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(self.ni))
                pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, x in pbar:
                    if cache == "disk":
                        b += self.npy_files[i].stat().st_size
                    else:  # 'ram'
                        (
                            self.ims[i],
                            self.im_hw0[i],
                            self.im_hw[i],
                        ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                        b += self.ims[i].nbytes
                    pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {cache})"
                pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(sum(self.ni.values()), 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = (
            b * sum(self.ni.values()) / n * (1 + safety_margin)
        )  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = (
            mem_required < mem.available
        )  # to cache or not to cache, that is the question
        if not cache:
            logger.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int)
            * self.stride
        )
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        images_and_labels = self.get_grouped_image_and_label(index)
        return [
            self.transforms(image_and_label) for image_and_label in images_and_labels
        ]

    def get_grouped_image_and_label(self, index) -> List[Dict[str, any]]:
        """
        Extend to return a list of dictionaries instead of a single dictionary
        """
        """Get and return label information from the dataset."""
        prefix = self.prefix_list[index]
        labels = deepcopy(
            self.grouped_labels[prefix]
        )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        res = []
        for group_index, label in enumerate(labels):
            label.pop("shape", None)  # shape is for rect, remove it
            (
                label["img"],
                label["ori_shape"],
                label["resized_shape"],
            ) = self.load_image(prefix=prefix, group_index=group_index)
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )  # for evaluation
            if self.rect:
                label["rect_shape"] = self.batch_shapes[self.batch[group_index]]
            res.append(self.update_labels_info(label))
        return res

    def get_image_and_label(self, index) -> Dict[str, any]:
        """Get and return label information from the dataset."""
        label = deepcopy(
            self.labels[index]
        )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        (
            label["img"],
            label["ori_shape"],
            label["resized_shape"],
        ) = self.load_image_group(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.grouped_labels)

    def update_labels_info(self, label):
        """custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_grouped_labels(self):
        # should be 15148 labels in total
        self.grouped_label_files = grouped_img2label_paths(self.image_groups)
        cache_path = Path(
            list(self.grouped_label_files.values())[0][0]
        ).parent.with_suffix(".cache_grouped")
        try:
            cache, exists = (
                # shoul still work as before, instead of loading a dict we load a list of dicts
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.grouped_label_files) + get_hash(
                self.image_groups
            )
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = (
                self.cache_labels_grouped(cache_path),
                False,
            )  # run cache ops

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        grouped_labels = cache["grouped_labels"]
        # orig is a dict with a single key (labels) with a list of dicts (one element per sample)
        if not grouped_labels:
            logger.warning(
                f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        # self.im_files = [lb["im_file"] for lb in grouped_labels]  # update im_fi
        self.image_groups = {
            prefix: [sample["im_file"] for sample in gr]
            for prefix, gr in grouped_labels.items()
        }

        return grouped_labels

    def cache_labels_grouped(self, path=Path("./labels.cache_gouped")):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        res = {}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        # total = len(self.im_files)
        # total_by_group = dict(map(lambda x: len(x), self.image_groups.values()))
        total_by_group = {k: len(v) for k, v in self.image_groups.items()}
        assert not self.use_keypoints, "Not needed for our use case"
        nkpt, ndim = 0, 0
        for prefix, total in total_by_group.items():
            x = []
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(
                    func=verify_image_label,
                    iterable=zip(
                        self.image_groups[prefix],
                        self.grouped_label_files[prefix],
                        repeat(self.prefix),
                        repeat(self.use_keypoints),
                        repeat(len(self.data["names"])),
                        repeat(nkpt),
                        repeat(ndim),
                    ),
                )
                pbar = TQDM(results, desc=desc, total=total)
                for (
                    im_file,
                    lb,
                    shape,
                    segments,
                    keypoint,
                    nm_f,
                    nf_f,
                    ne_f,
                    nc_f,
                    msg,
                ) in pbar:
                    nm += nm_f
                    nf += nf_f
                    ne += ne_f
                    nc += nc_f
                    if im_file:
                        x.append(
                            dict(
                                im_file=im_file,
                                shape=shape,
                                cls=lb[:, 0:1],  # n, 1
                                bboxes=lb[:, 1:],  # n, 4
                                segments=segments,
                                keypoints=keypoint,
                                normalized=True,
                                bbox_format="xywh",
                            )
                        )
                    if msg:
                        msgs.append(msg)
                    pbar.desc = (
                        f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                    )
                pbar.close()
            res[prefix] = x
        res = {"grouped_labels": res}
        if msgs:
            logger.info("\n".join(msgs))
        if nf == 0:
            logger.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}.")
        res["hash"] = get_hash(self.grouped_label_files) + get_hash(self.image_groups)
        # res["results"] = nf, nm, ne, nc, len(self.im_files)
        res["results"] = nf, nm, ne, nc
        res["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, res)
        return res


class SequencedYOLODataset(SequencedFramesUltralyticsDataset):
    """ """

    def __init__(
        self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs
    ):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (
            self.use_segments and self.use_keypoints
        ), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop("bboxes")
        segments = label.pop("segments")
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        label["instances"] = Instances(
            bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized
        )
        return label

    @staticmethod
    def collate_fn(batch):
        """
        Reminder: Limited to BS = 1
        ."""
        new_batch = []
        for frame in batch[0]:
            new_frame = {}
            values = list(frame.values())
            keys = frame.keys()
            for i, k in enumerate(keys):
                value = values[i]
                if k == "img":
                    value = value.unsqueeze(dim=0)
                # if k in ["masks", "keypoints", "bboxes", "cls"]:
                #     value = torch.cat(value, 0)
                if k in ["ori_shape", "ratio_pad"]:
                    new_frame[k] = [value]
                elif k == "im_file":
                    new_frame[k] = [value.__str__()]
                else:
                    new_frame[k] = value
            # to find which index corresponds to what image in a concatenated labels tensor
            new_frame["batch_idx"] = torch.zeros(frame["bboxes"].shape[0])
            # for i in range(len(new_frame["batch_idx"])):
            #     new_frame["batch_idx"][i] += i  # add target image index for build_targets()
            # new_frame["batch_idx"] = torch.cat(new_frame["batch_idx"], 0)
            new_batch.append(new_frame)
        return new_batch


class FramesYOLODataset(FramesUltralyticsDataset):
    """ """

    def __init__(
        self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs
    ):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (
            self.use_segments and self.use_keypoints
        ), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop("bboxes")
        segments = label.pop("segments")
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        label["instances"] = Instances(
            bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized
        )
        return label

    @staticmethod
    def collate_fn(batch):
        """
        Reminder: Limited to BS = 1
        ."""
        new_batch = []
        for frame in batch[0]:
            new_frame = {}
            values = list(frame.values())
            keys = frame.keys()
            for i, k in enumerate(keys):
                value = values[i]
                if k == "img":
                    value = value.unsqueeze(dim=0)
                # if k in ["masks", "keypoints", "bboxes", "cls"]:
                #     value = torch.cat(value, 0)
                if k in ["ori_shape", "ratio_pad"]:
                    new_frame[k] = [value]
                elif k == "im_file":
                    new_frame[k] = [value.__str__()]
                else:
                    new_frame[k] = value
            # to find which index corresponds to what image in a concatenated labels tensor
            new_frame["batch_idx"] = torch.zeros(frame["bboxes"].shape[0])
            # for i in range(len(new_frame["batch_idx"])):
            #     new_frame["batch_idx"][i] += i  # add target image index for build_targets()
            # new_frame["batch_idx"] = torch.cat(new_frame["batch_idx"], 0)
            new_batch.append(new_frame)
        return new_batch
