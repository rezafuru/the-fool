import os
import shutil
from pathlib import Path


def merge_datasets(src1, src2, dest, prefix1, prefix2):
    # Ensure destination directories exist
    os.makedirs(os.path.join(dest, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dest, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(dest, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dest, "images", "val"), exist_ok=True)

    # Copy and rename files from source1
    for split in ["train", "val"]:
        for kind in ["labels", "images"]:
            src_dir = os.path.join(src1, kind, split)
            dest_dir = os.path.join(dest, kind, split)

            for fname in os.listdir(src_dir):
                base, ext = os.path.splitext(fname)
                new_fname = prefix1 + base + ext
                shutil.copy2(
                    os.path.join(src_dir, fname), os.path.join(dest_dir, new_fname)
                )

    # Copy and rename files from source2
    for split in ["train", "val"]:
        for kind in ["labels", "images"]:
            src_dir = os.path.join(src2, kind, split)
            dest_dir = os.path.join(dest, kind, split)

            for fname in os.listdir(src_dir):
                base, ext = os.path.splitext(fname)
                new_fname = prefix2 + base + ext
                shutil.copy2(
                    os.path.join(src_dir, fname), os.path.join(dest_dir, new_fname)
                )


if __name__ == "__main__":
    dataset_root = Path(os.path.expanduser("~/resources/datasets"))
    src1 = dataset_root / "xView_tiled_filtered_512x512"
    src2 = dataset_root / "DOTA-2_tiled_filtered_512x512"
    dest = dataset_root / "xView_DOTA_tiled_filtered_512x512"
    prefix1 = "D2_"
    prefix2 = "xV_"

    merge_datasets(src1, src2, dest, prefix1, prefix2)
    print("Datasets merged successfully!")
