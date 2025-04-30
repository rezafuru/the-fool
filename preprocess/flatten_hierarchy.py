import argparse
import os
import shutil


def flatten_directory(root_dir):
    # Move files to root directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            original_file_path = os.path.join(root, file)
            if root != root_dir:  # Avoid moving files already in the root directory
                new_file_path = os.path.join(root_dir, file)
                shutil.move(original_file_path, new_file_path)

    # Delete all subdirectories
    for root, dirs, _ in os.walk(root_dir):
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))


if __name__ == "__main__":
    dataset_root = os.path.expanduser("~/resources/datasets")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()
    root_directory = os.path.join(dataset_root, args.root_dir)
    flatten_directory(root_directory)
