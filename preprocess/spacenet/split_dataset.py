import os
import shutil
import random
import sys


def split_dataset(root_dir, output_dir, percentage):
    # Define the paths for images and labels
    images_train_path = os.path.join(root_dir, 'images', 'train')
    labels_train_path = os.path.join(root_dir, 'labels', 'train')

    # Define output paths
    output_images_train = os.path.join(output_dir, 'images', 'train')
    output_labels_train = os.path.join(output_dir, 'labels', 'train')
    output_images_val = os.path.join(output_dir, 'images', 'val')
    output_labels_val = os.path.join(output_dir, 'labels', 'val')

    # Ensure output directories exist
    for path in [output_images_train, output_labels_train, output_images_val, output_labels_val]:
        os.makedirs(path, exist_ok=True)

    # List all image files
    images = [f for f in os.listdir(images_train_path) if f.endswith('.tif')]
    random.shuffle(images)

    # Calculate the split index
    val_count = int(len(images) * (percentage / 100))
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Copy files
    for img in train_images:
        if "1831" in img:
            continue
        shutil.copy(os.path.join(images_train_path, img), output_images_train)
        try:
            shutil.copy(os.path.join(labels_train_path, img.replace('.tif', '.txt')), output_labels_train)
        except Exception as e:
            print(e)

    for img in val_images:
        shutil.copy(os.path.join(images_train_path, img), output_images_val)
        try:
            shutil.copy(os.path.join(labels_train_path, img.replace('.tif', '.txt')), output_labels_val)
        except Exception as e:
            print(e)



    print(f"Processed {len(train_images)} training and {len(val_images)} validation samples.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <root_dir> <output_dir> <percentage>")
        sys.exit(1)

    root_dir = os.path.expanduser(sys.argv[1])
    output_dir = os.path.expanduser(sys.argv[2])
    percentage = float(sys.argv[3])

    split_dataset(root_dir, output_dir, percentage)
