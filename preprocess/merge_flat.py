import os
import shutil
from pathlib import Path
import argparse


def copy_files_from_sources_to_destination(source_dirs, dest_dir):
    """
    Copy all files from multiple source directories to a destination directory.

    Parameters:
    - source_dirs (list): List of source directories.
    - dest_dir (str): Destination directory.
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for source_dir in source_dirs:
        # Walk through the source directory
        for dirpath, dirnames, filenames in os.walk(source_dir):
            for filename in filenames:
                # Construct full file path
                source_file_path = os.path.join(dirpath, filename)
                dest_file_path = os.path.join(dest_dir, filename)

                # Handle potential file name collisions
                counter = 1
                while os.path.exists(dest_file_path):
                    name, ext = os.path.splitext(filename)
                    dest_file_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1

                # Copy the file
                shutil.copy2(source_file_path, dest_file_path)


if __name__ == "__main__":
    # Example usage
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
    source_directories = [dataset_root / args.path1, dataset_root / args.path2]
    # destination_directory = datset_root / "DOTA-xView-2/images/val"
    destination_directory = dataset_root / args.dest
    copy_files_from_sources_to_destination(source_directories,
                                           destination_directory)
