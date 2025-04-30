import os


def create_empty_label_files(dataset_root, mode):
    """
    Create empty .txt label files for images without corresponding label files.

    Parameters:
    - dataset_root: the root path of the dataset
    - mode: either "train" or "val"
    """
    # Define the image and label directories based on the mode
    image_dir = os.path.join(dataset_root, "images", mode)
    label_dir = os.path.join(dataset_root, "labels", mode)

    # Make sure the directories exist
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} does not exist!")
        return
    if not os.path.exists(label_dir):
        print(f"Error: {label_dir} does not exist!")
        return

    # Iterate over each file in the image directory
    for filename in os.listdir(image_dir):
        name, ext = os.path.splitext(filename)
        # If the corresponding label file doesn't exist, create an empty one
        if not os.path.exists(os.path.join(label_dir, name + ".txt")):
            with open(os.path.join(label_dir, name + ".txt"), "w") as f:
                pass
            print(f"Created empty label for {name}")


# Usage
dataset_root = (
    os.path.expanduser("~/resources/datasets/coco_ultralytics")
    # Replace this with the path to your dataset root
)
create_empty_label_files(dataset_root, "train2017")
create_empty_label_files(dataset_root, "val2017")
