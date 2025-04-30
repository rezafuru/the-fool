import argparse
import math
import os
import cv2
import numpy as np
from tqdm import tqdm


def count_intensity_pixels(image: np.ndarray) -> (int, int):
    """
    Count the number of pixels in the image that have zero intensity and non-zero intensity.

    Args:
    - image (np.ndarray): A 3D OpenCV image array.

    Returns:
    - int: The number of pixels with zero intensity.
    - int: The number of pixels with non-zero intensity.
    """
    # Check that the image is indeed a 3D numpy array with 3 channels
    # if len(image.shape) != 3 or image.shape[2] != 3:
    #     raise ValueError("Input image must be a 3D numpy array with 3 channels.")

    # Create a mask where all channels are 0 for each pixel
    zero_intensity_mask = np.all(image == 0, axis=2)

    # Count the number of True values in the mask (zero intensity)
    zero_count = np.sum(zero_intensity_mask)

    # Count the number of non-zero intensity pixels
    non_zero_count = (
        image.size // 3 - zero_count
    )  # total pixels - zero intensity pixels

    return zero_count, non_zero_count


def pad_image(img, tile_size):
    h, w = img.shape[:2]
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)


def split_image(img, tile_size):
    h, w = img.shape[:2]
    return [
        img[y : y + tile_size, x : x + tile_size]
        for y in range(0, h, tile_size)
        for x in range(0, w, tile_size)
    ]



def main(
    original_data_root, tiled_data_root, tile_size, threshold_nonzero_pixels=256 * 256
):
    # img_dirs = ["images/train", "images/val"]
    img_dirs = [""]

    tiles_info = []
    no_filtered_tiles = 0
    # Loops input dirs
    for img_dir in img_dirs:
        orig_img_path = os.path.join(original_data_root, img_dir)

        tiled_img_path = os.path.join(tiled_data_root, "images/train" if img_dir == "" else img_dir)

        os.makedirs(tiled_img_path, exist_ok=True)

        # Loops images
        for img_name in tqdm(os.listdir(orig_img_path)):
            img = cv2.imread(os.path.join(orig_img_path, img_name))
            assert img is not None, f"Failed to read image {img_name}"
            padded_img = pad_image(img, tile_size)

            tiles = split_image(padded_img, tile_size)

            base_name = os.path.splitext(img_name)[0]
            tile_names = []
            label_name = base_name + ".txt"

            # Loops created tiles
            for idx, tile in enumerate(tiles):
                tile_name = f"{base_name}_{idx}.png"
                zero_count, non_zero_pixels = count_intensity_pixels(tile)
                if non_zero_pixels >= threshold_nonzero_pixels:
                    tile_names.append(tile_name)
                    cv2.imwrite(os.path.join(tiled_img_path, tile_name), tile)
                else:
                    no_filtered_tiles += 1

            tiles_info.append(" ".join(tile_names))

    with open(os.path.join(tiled_data_root, "tiles_info.txt"), "w") as f:
        f.write("\n".join(tiles_info))
    print(40 * "-")
    print(f"Filtered: {no_filtered_tiles} background tiles")
    print(40 * "-")


if __name__ == "__main__":
    dataset_root = os.path.expanduser("~/resources/datasets/")

    parser = argparse.ArgumentParser(
        description="Merge two datasets without any nested hierarchy"
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--img_prefix", default="", required=False, type=str)

    # parser.add_argument("--group_size", type=int, required=True)
    parser.add_argument("--tile_size", type=int, required=True)
    parser.add_argument("--threshold_zero_dim", required=False, type=int, default=None)
    parser.add_argument("--input_root", required=False, type=str, default=None)
    parser.add_argument("--output_root", required=False, type=str, default=None)
    args = parser.parse_args()
    output_root = os.path.expanduser(args.output_root) if args.output_root else None
    input_root = os.path.expanduser(args.input_root) if args.input_root else None
    # dataset_name = "DOTA-xView-2"
    dataset_name = args.dataset
    input_root = (
        (f"{dataset_root}/{dataset_name}")  # Replace with your input dataset root path
        if input_root is None
        else f"{input_root}/{dataset_name}"
    )
    tile_size = args.tile_size  # Replace with your tile size
    # group_size = args.group_size
    # Replace with your output dataset root path
    threshold_zero_pixels = (
        args.threshold_zero_dim * args.threshold_zero_dim
        if args.threshold_zero_dim
        else None
    )
    # if "xView" in dataset_name:
    #     dataset_name = "xView"
    img_prefix = args.img_prefix
    output_root = (
        f"{dataset_root}/{dataset_name}_tiled_filtered_{tile_size}x{tile_size}"
        if output_root is None
        else f"{output_root}/{dataset_name}_tiled_filtered_{tile_size}x{tile_size}"
    )
    main(
        input_root,
        output_root,
        tile_size,
        threshold_nonzero_pixels=threshold_zero_pixels,
    )
