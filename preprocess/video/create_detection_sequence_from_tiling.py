import argparse
import glob
from os.path import basename
from pathlib import Path

from PIL import Image
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import shutil
import random
import numpy as np
from tqdm import tqdm
from ultralytics.utils.checks import check_version
from albumentations import (
    Compose,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
)


def ultralytics_to_voc(image, labels):
    h, w = image.shape[:2]
    pascal_boxes = []
    class_labels = []
    for label in labels:
        class_label, cx, cy, bw, bh = label
        x_min = int((cx - bw / 2) * w)
        x_max = int((cx + bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        y_max = int((cy + bh / 2) * h)
        pascal_boxes.append([x_min, y_min, x_max, y_max])
        class_labels.append(int(class_label))
    return pascal_boxes, class_labels


def voc_to_ultralytics(augmented_boxes):
    augmented_ultralytics_boxes = []
    img = augmented_boxes["image"]
    h, w = img.shape[:2]
    bboxes = augmented_boxes["bboxes"]
    class_labels = augmented_boxes["class_labels"]
    for cls_label, box in zip(class_labels, bboxes):
        x_min, y_min, x_max, y_max = box
        cx = ((x_max + x_min) / 2) / w
        cy = ((y_max + y_min) / 2) / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        augmented_ultralytics_boxes.append([cls_label, cx, cy, bw, bh])

    return augmented_ultralytics_boxes


# class Albumentations:
#     """Albumentations transformations. Optional, uninstall package to disable.
#     Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive Histogram Equalization,
#     random change of brightness and contrast, RandomGamma and lowering of image quality by compression."""
#
#     def __init__(self, p=1.0):
#         """Initialize the transform object for YOLO bbox formatted params."""
#         self.p = p
#         self.transform = None
#         try:
#             import albumentations as A
#
#             # check_version(A.__version__, '1.0.3', hard=True)  # version requirement
#
#             T = [
#                 A.Blur(p=1.01),
#                 A.MedianBlur(p=1.01),
#                 A.ToGray(p=1.01),
#                 A.CLAHE(p=1.01),
#                 A.RandomBrightnessContrast(p=1.0),
#                 A.RandomGamma(p=1.0),
#             ]
#             T = A.OneOf(T)# transforms
#             self.transform = A.Compose(T, bbox_params={'format': 'pascal_voc', 'label_fields': ['class_labels']})
#
#         except ImportError:  # package not installed, skip
#             pass
#         except Exception as e:
#             print(e)
#             # print(f'{prefix}{e}')

# from albumentations.pytorch import ToTensorV
augmented_paths = []


def sep_label_from_bbox(bboxes):
    res_bboxes = []
    labels = []
    for bbox in bboxes:
        labels.append(bbox[0])
        res_bboxes.append(bbox[1:])

    return res_bboxes, labels


def merge_label_to_bboxes(bboxes, labels):
    res = []
    for bbox, label in zip(bboxes, labels):
        res.append([label] + bbox)
    return res


# def augment_image_and_label(image, label):
#     # Convert PIL image to NumPy array
#     image_np = np.array(image)
#     # if len(label) == 0:
#     #     return image, label
#     # Define augmentation
#     # transformations = Compose(
#     #     [
#     #         OneOf(
#     #             [
#     #                 ElasticTransform(
#     #                     p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
#     #                 ),
#     #                 GridDistortion(p=0.5),
#     #                 OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.1),
#     #             ],
#     #             p=0.8,
#     #         ),
#     #         Flip(p=0.5),
#     #         Transpose(p=0.5),
#     #     ],
#     #     bbox_params={
#     #         "format": "coco",
#     #         "label_fields": ["class_labels"],
#     #         "min_area": 0.1,
#     #         "min_visibility": 0.5,
#     #     },
#     # )
#     voc_bboxes, cls_labels = ultralytics_to_voc(image_np, label)
#     augmented = aug.transform(
#         image=image_np, bboxes=voc_bboxes, class_labels=cls_labels
#     )
#     augmented_img = augmented["image"]
#
#     transformed = voc_to_ultralytics(augmented)
#     transformed_filtered = []
#     # Check for invalid bboxes
#     for bbox in transformed:
#         x_min, y_min, x_max, y_max = bbox[:4]
#         if x_max > x_min and y_max > y_min:
#             transformed_filtered.append(bbox)
#         else:
#             print(bbox)
#
#     # Convert NumPy array back to PIL image
#     transformed_image = Image.fromarray(np.uint8(augmented_img), "RGB")
#
#     # return transformed_image, transformed_bboxes
#     return transformed_image, []


def augment(image):
    image_np = np.array(image)
    transformations = Compose(
        [
            OneOf(
                [
                    RandomBrightnessContrast(
                        p=1.0, brightness_limit=0.2, contrast_limit=0.2
                    ),
                    RandomGamma(p=1.0, gamma_limit=(80, 120)),
                    HueSaturationValue(
                        p=1.0,
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                    ),
                ],
                p=1,
            )
        ]
    )
    transformed = transformations(image=image_np)
    return Image.fromarray(transformed["image"].astype("uint8"), "RGB")


def save_files(
    file_paths,
    sequence,
    group_idx,
    dest_path,
    augment_count,
    is_label=False,
    add_augment_to_orig=False,
):
    global augmented_paths
    group_folder = os.path.join(dest_path, sequence, f"group{group_idx}")
    os.makedirs(group_folder, exist_ok=True)
    for idx, src_path in enumerate(file_paths):
        # os.path.join(group_folder, f"{Path(group_folder).stem}_{os.path.basename(src_path)}")
        _src_path = Path(src_path)
        # dest_file_path = os.path.join(group_folder, os.path.basename(src_path))
        _group_folder = Path(group_folder)
        file_name = f"{_group_folder.parent.stem}_g{group_idx}_{_src_path.stem.split('_')[1]}{_src_path.suffix}"
        dest_file_path = Path(group_folder) / file_name
        # dest_file_path = os.path.join(group_folder, f"{_src_path.stem}_{Path(group_folder).stem}{_src_path.suffix}")
        if is_label:
            shutil.copy(src_path, dest_file_path)
        else:
            shutil.copy(src_path, dest_file_path)

    if augment_count > 0:
        last_file = file_paths[-1]
        last_idx = int(os.path.basename(last_file).split("_")[1].split(".")[0])
        for i in range(1, augment_count + 1):
            new_idx = last_idx + i
            new_file_name = f"{sequence}_g{group_idx}_{new_idx}"

            if is_label:
                new_file_name_orig = f"{sequence}_{new_idx}.txt"
                new_file_path = os.path.join(group_folder, f"{new_file_name}.txt")
                shutil.copy(last_file, new_file_path)
                if add_augment_to_orig:
                    shutil.copy(last_file, Path(last_file).parent / new_file_name_orig)
            else:
                new_file_name_orig = f"{sequence}_{new_idx}.png"
                new_file_path = os.path.join(group_folder, f"{new_file_name}.png")
                image = Image.open(last_file)
                augmented_image = augment(image)
                augmented_image.save(new_file_path)
                augmented_paths.append(new_file_path)
                if add_augment_to_orig:
                    augmented_image.save(Path(last_file).parent / new_file_name_orig)


def main(source_path, n, dest_path, add_augment_to_orig=False):
    for folder_type in ["train", "val"]:
        for data_type in ["images", "labels"]:
            src_folder = os.path.join(source_path, data_type, folder_type)
            if n != 5:
                dest_folder = os.path.join(
                    dest_path, data_type, f"{folder_type}_sequenced_n={n}"
                )
            else:
                dest_folder = os.path.join(
                    dest_path, data_type, f"{folder_type}_sequenced"
                )

            sequences = set()
            for f in glob.glob(
                f"{src_folder}/*.txt"
                if data_type == "labels"
                else f"{src_folder}/*.png"
            ):
                sequence = os.path.basename(f).split("_")[0]
                sequences.add(sequence)

            for sequence in tqdm(sorted(list(sequences))):
                sequence_files = sorted(
                    glob.glob(f"{src_folder}/{sequence}_*"),
                    key=lambda x: int(basename(x).split("_")[1].split(".")[0]),
                )
                # for file in sequence_files:
                #     try:
                #         index = int(file.split("_")[1].split(".")[0])
                #     except ValueError:
                #         print(f"Error converting file name to integer: {file}")
                # sequence_files = sorted(
                #     sequence_files, key=lambda x: int(x.split("_")[1].split(".")[0])
                # )

                num_files = len(sequence_files)
                num_groups = num_files // n
                remaining = num_files % n

                idx = 0
                for group_idx in range(num_groups):
                    group_files = sequence_files[idx : idx + n]
                    idx += n
                    save_files(
                        group_files,
                        sequence,
                        group_idx,
                        dest_folder,
                        0,
                        is_label=(data_type == "labels"),
                        add_augment_to_orig=add_augment_to_orig,
                    )

                if remaining > 0:
                    last_group_files = sequence_files[idx:]
                    save_files(
                        last_group_files,
                        sequence,
                        num_groups,
                        dest_folder,
                        n - remaining,
                        is_label=(data_type == "labels"),
                        add_augment_to_orig=add_augment_to_orig,
                    )

    print("Augmented files:")
    for path in augmented_paths:
        print(path)


if __name__ == "__main__":
    dataset_root = os.path.expanduser("~/resources/datasets")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", required=True, type=str)
    # parser.add_argument("--img_prefix", default="", required=False, type=str)
    parser.add_argument("--group_size", type=int, required=True)
    # parser.add_argument("--threshold_zero_dim", required=False, type=int, default=None)
    parser.add_argument("--input_root", required=False, type=str, default=None)
    parser.add_argument("--output_root", required=False, type=str, default=None)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--add_augment_to_orig", action="store_true")
    args = parser.parse_args()
    dataset_name = args.dataset
    input_root = os.path.expanduser(args.input_root) if args.input_root else None
    output_root = os.path.expanduser(args.output_root) if args.output_root else None
    input_root = (
        (f"{dataset_root}/{dataset_name}")  # Replace with your input dataset root path
        if input_root is None
        else f"{input_root}/{dataset_name}"
    )
    group_size = args.group_size
    # Replace with your output dataset root path
    if args.inplace:
        output_root = input_root
    else:
        output_root = (
            f"{dataset_root}/{dataset_name}_sequenced_n={group_size}"
            if output_root is None
            else f"{output_root}/{dataset_name}_sequenced_n={group_size}"
        )
    print("input root: ", input_root)
    print("output root: ", output_root)
    main(
        source_path=input_root,
        n=group_size,
        dest_path=output_root,
        add_augment_to_orig=args.add_augment_to_orig,
    )
