import random
import shutil
from pathlib import Path

from misc.util import mkdir


def split_array(arr, percentage):
    split_index = int(len(arr) * (percentage / 100.0))
    return arr[:split_index], arr[split_index:]


root_dataset = Path("/home/alireza/resources/datasets/NWPU-VHR-10/")
root_orig = root_dataset / "merged"
root_split = root_dataset / "split"

train_split = root_split / "train"
test_split = root_split / "test"

mkdir(train_split.__str__())
mkdir(test_split.__str__())

labels_orig = root_dataset / "labels_orig"
labels_split_orig_train = root_split / "labels_orig_train"
labels_split_orig_test = root_split / "labels_orig_test"
labels_orig_ultralytics = root_dataset / "labels_ultralytics"
labels_split_ultralytics_train = root_split / "labels_ultralytics_train"
labels_split_ultralytics_test = root_split / "labels_ultralytics_test"

mkdir(root_split.__str__())
mkdir(labels_split_ultralytics_train.__str__())
mkdir(labels_split_ultralytics_test.__str__())
mkdir(labels_split_orig_train.__str__())
mkdir(labels_split_orig_test.__str__())
mkdir(root_split.__str__())
dir_dataset = root_orig / "dataset"

paths_dataset = list(dir_dataset.iterdir())
random.shuffle(paths_dataset)

train_set, test_set = split_array(paths_dataset, 90)

for sample_path in train_set:
    sample_base_name = sample_path.stem
    idx_sample = int(sample_base_name)
    sample_img_name = f"{sample_base_name}.jpg"
    sample_label_name = f"{sample_base_name}.txt"

    sample_path_split_img = train_split / sample_img_name
    shutil.copy(src=sample_path.__str__(), dst=sample_path_split_img)

    if (
        idx_sample <= 650
    ):  # we assigned indexes to negative samples from len(positives) + 1 on
        orig_label_path = labels_orig / sample_label_name
        orig_ultralytics_path = labels_orig_ultralytics / sample_label_name

        split_label_path = labels_split_orig_train / sample_label_name
        split_label_path_ultralytics = (
            labels_split_ultralytics_train / sample_label_name
        )

        shutil.copy(src=orig_label_path.__str__(), dst=split_label_path.__str__())
        shutil.copy(
            src=orig_ultralytics_path.__str__(),
            dst=split_label_path_ultralytics.__str__(),
        )

for sample_path in test_set:
    sample_base_name = sample_path.stem
    idx_sample = int(sample_base_name)
    sample_img_name = f"{sample_base_name}.jpg"
    sample_label_name = f"{sample_base_name}.txt"

    sample_path_split_img = test_split / sample_img_name
    shutil.copy(src=sample_path.__str__(), dst=sample_path_split_img)

    if (
        idx_sample <= 650
    ):  # we assigned indexes to negative samples from len(positives) + 1 on
        orig_label_path = labels_orig / sample_label_name
        orig_ultralytics_path = labels_orig_ultralytics / sample_label_name

        split_label_path = labels_split_orig_test / sample_label_name
        split_label_path_ultralytics = labels_split_ultralytics_test / sample_label_name

        shutil.copy(src=orig_label_path.__str__(), dst=split_label_path.__str__())
        shutil.copy(
            src=orig_ultralytics_path.__str__(),
            dst=split_label_path_ultralytics.__str__(),
        )
