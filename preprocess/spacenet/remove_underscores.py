import os
from pathlib import Path

def remove_underscores(directory):
    # Iterate over all files in the given directory, including subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if 'img' in filename:
                # Create a Path object for the original file
                original_file = Path(root) / filename
                # Create a new filename by removing all underscores
                new_filename = filename.replace('img', '')
                # Create a Path object for the new file
                new_file = Path(root) / new_filename
                # Rename the file
                original_file.rename(new_file)
                print(f"Renamed '{original_file}' to '{new_file}'")

# Replace 'your_directory_path_here' with the path to your target directory

remove_underscores(os.path.expanduser("~/resources/datasets/SpaceNet-3-OD-custom-split"))
