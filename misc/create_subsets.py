import os
from pathlib import Path

dataset_root = os.environ.get("DATASET_ROOT")
assert dataset_root and os.path.isdir(
    os.path.expanduser(dataset_root)
), "ILSVRC_ROOT is not set or does not exist"
subsets = {
    # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    # todo
}

dst_root = Path("../resources/datasets")
dataset_root = Path(dataset_root)

train_dir = os.listdir(dataset_root / "train")
val_dir = os.listdir(dataset_root / "val")
train_dir.sort(), val_dir.sort()

for subset, class_idxs in subsets.items():
    dst_subset = dst_root / subset
    dst_train = dst_subset / "train"
    dst_val = dst_subset / "val"
    for class_idx in class_idxs:
        class_dir_name = train_dir[class_idx]
        assert class_dir_name == val_dir[class_idx]
        src_train = dataset_root / "train" / class_dir_name
        src_val = dataset_root / "val" / class_dir_name
        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val, exist_ok=True)
        os.system(f"cp -r {src_train} {dst_train}")
        os.system(f"cp -r {src_val} {dst_val}")
