import argparse
import random
import shutil
from pathlib import Path

import cv2
import os
import numpy as np


def count_intensity_pixels(image: np.ndarray) -> (int, int):
    zero_pixels = np.sum((image == 0).all(axis=2))
    non_zero_pixels = image.shape[0] * image.shape[1] - zero_pixels
    return zero_pixels, non_zero_pixels


def pad_image(img, tile_size):
    h, w = img.shape[:2]
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def tile_image(
    image_path,
    tile_size,
    output_dir,
    group_size,
    threshold_zero_pixels,
    aug_for_remainders,
    img_prefix="",
):
    img = cv2.imread(image_path)
    img = pad_image(img, tile_size)
    img_name = os.path.basename(image_path).split(".")[0]
    h, w, _ = img.shape
    tile_count = 0
    group_count = 0
    group = []
    remainder_groups = []
    last_tile = None
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = img[i : i + tile_size, j : j + tile_size]
            zero_count, _ = count_intensity_pixels(tile)
            if zero_count <= threshold_zero_pixels or threshold_zero_pixels is None:
                group_dir = os.path.join(
                    output_dir, f"{img_prefix}{img_name}/group{group_count}"
                )
                os.makedirs(group_dir, exist_ok=True)
                tile_path = os.path.join(group_dir, f"{img_name}_{tile_count}.png")
                cv2.imwrite(tile_path, tile)
                group.append(f"{img_name}/group{group_count}")
                tile_count += 1
                last_tile = tile
                if tile_count % group_size == 0:
                    group_count += 1
                    tile_count = 0

    # Perform augmentation to fill remaining spots
    if aug_for_remainders:
        while 0 < tile_count < group_size:
            angle = random.randint(15, 270)
            rotated_tile = rotate_image(last_tile, angle)
            tile_path = os.path.join(
                group_dir, f"{img_name}_g{group_count}_{tile_count}.png"
            )
            cv2.imwrite(tile_path, rotated_tile)
            tile_count += 1

    if tile_count != 0:
        remainder_groups.append(f"{img_name}/group{group_count}")
    return group, remainder_groups


def main(
    input_root,
    tile_size,
    group_size,
    output_root,
    threshold_zero_pixels,
    aug_for_remainders,
    img_prefix="",
):
    train_list = []
    val_list = []
    remainder_groups = []
    for subset in [""]:
        # for subset in ["images/val", "images/train"]:
        input_dir = os.path.join(input_root, subset)
        if subset == "":
            output_dir = os.path.join(output_root, "images/train_sequenced")
        else:
            output_dir = os.path.join(output_root, subset).replace(subset, "sequenced/")
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            groups, remainders = tile_image(
                img_path,
                tile_size,
                output_dir,
                group_size,
                threshold_zero_pixels,
                aug_for_remainders,
                img_prefix,
            )
            remainder_groups.extend(remainders)
            if subset == "images/val":
                val_list.extend(groups)
            else:
                train_list.extend(groups)

        with open(os.path.join(output_root, "train.list"), "w") as f:
            f.write("\n".join(train_list))

        with open(os.path.join(output_root, "val.list"), "w") as f:
            f.write("\n".join(val_list))

    print("Groups with remainders:", remainder_groups)
    print("Size:", len(remainder_groups))


if __name__ == "__main__":
    dataset_root = os.path.expanduser("~/resources/datasets/")
    parser = argparse.ArgumentParser(
        description="Merge two datasets without any nested hierarchy"
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--img_prefix", default="", required=False, type=str)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument("--tile_size", type=int, required=True)
    parser.add_argument("--threshold_zero_dim", required=False, type=int, default=None)
    parser.add_argument("--input_root", required=False, type=str, default=None)
    parser.add_argument("--output_root", required=False, type=str, default=None)

    # parser.add_argument("--dest", required=True)
    # path1= datset_root / "DOTA-2/images/val"
    # path2 = datset_root / "xView/images/val"
    args = parser.parse_args()
    output_root = os.path.expanduser(args.output_root) if args.output_root else None
    input_root = os.path.expanduser(args.input_root) if args.input_root else None
    # dataset_name = "DOTA-xView-2"
    dataset_name = args.dataset
    input_root = (
        (f"{dataset_root}/{dataset_name}")  # Replace with your input dataset root path
        if input_root is None
        else f"{input_root}/{dataset_name}"
    )
    tile_size = args.tile_size  # Replace with your tile size
    group_size = args.group_size
    # Replace with your output dataset root path
    threshold_zero_pixels = (
        args.threshold_zero_dim * args.threshold_zero_dim
        if args.threshold_zero_dim
        else None
    )
    img_prefix = args.img_prefix
    output_root = (
        f"{dataset_root}/{dataset_name}_tiled_filtered_{tile_size}x{tile_size}"
        if output_root is None
        else f"{output_root}/{dataset_name}_tiled_filtered_{tile_size}x{tile_size}"
    )

    print("tile_size:", tile_size)
    print("group_size:", group_size)
    print("Threshold zero pixels:", threshold_zero_pixels)
    print("Output root:", output_root)
    print("input_root:", input_root)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    main(
        input_root,
        tile_size,
        group_size,
        output_root,
        threshold_zero_pixels,
        aug_for_remainders=True,
        img_prefix=img_prefix,
    )
