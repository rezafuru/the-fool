import argparse
import os
import shutil
from pathlib import Path


# def assert_structure(dir1, dir2):
#     dir1_structure = []
#     dir2_structure = []
#
#     for root, dirs, _ in os.walk(dir1):
#         rel_root = os.path.relpath(root, dir1)
#         for dir_name in dirs:
#             dir1_structure.append(os.path.join(rel_root, dir_name))
#
#     for root, dirs, _ in os.walk(dir2):
#         rel_root = os.path.relpath(root, dir2)
#         for dir_name in dirs:
#             dir2_structure.append(os.path.join(rel_root, dir_name))
#
#     return set(dir1_structure) == set(dir2_structure)
#
#
# def merge_directories(dir1, dir2, output_dir):
#     if not assert_structure(dir1, dir2):
#         raise ValueError("Directory structures are not the same")
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for dir_path in [dir1, dir2]:
#         for root, _, files in os.walk(dir_path):
#             rel_root = os.path.relpath(root, dir_path)
#             target_root = os.path.join(output_dir, rel_root)
#
#             if not os.path.exists(target_root):
#                 os.makedirs(target_root)
#
#             for file in files:
#                 src_file = os.path.join(root, file)
#                 dst_file = os.path.join(target_root, file)
#                 shutil.copy2(src_file, dst_file)
def merge_directories(src1, src2, dst):
    # Create the output directory if it doesn't exist
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Merge contents of src1 into dst
    for item in os.listdir(src1):
        src1_item = os.path.join(src1, item)
        dst_item = os.path.join(dst, item)
        src2_item = os.path.join(src2, item)

        if os.path.isdir(src1_item):
            if not os.path.exists(dst_item):
                shutil.copytree(src1_item, dst_item)
            elif os.path.exists(src2_item):
                merge_directories(src1_item, src2_item, dst_item)
        else:
            shutil.copy2(src1_item, dst_item)

    # Merge contents of src2 into dst (overwriting duplicates)
    for item in os.listdir(src2):
        src2_item = os.path.join(src2, item)
        dst_item = os.path.join(dst, item)
        src1_item = os.path.join(src1, item)

        if os.path.isdir(src2_item):
            if not os.path.exists(dst_item):
                shutil.copytree(src2_item, dst_item)
            elif os.path.exists(src1_item):
                merge_directories(src1_item, src2_item, dst_item)
        else:
            shutil.copy2(src2_item, dst_item)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two datasets without any nested hierarchy")
    dataset_root = Path(os.path.expanduser("~/resources/datasets"))
    parser.add_argument(
        "--path1", required=True
    )
    parser.add_argument(
        "--path2", required=True
    )
    parser.add_argument("--dest", required=True)
    # path1= datset_root / "DOTA-2/images/val"
    # path2 = datset_root / "xView/images/val"
    args = parser.parse_args()
    dir1, dir2 = [dataset_root / args.path1, dataset_root / args.path2]

    # destination_directory = datset_root / "DOTA-xView-2/images/val"
    destination_directory = dataset_root / args.dest
    merge_directories(dir1, dir2, dst=destination_directory)