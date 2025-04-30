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


def split_labels(
    label_path, tile_size, num_tiles, img_width, img_height, area_threshold
):

    if not os.path.exists(label_path):
        with open(label_path, "a"):
            pass

    with open(label_path, "r") as f:
        lines = f.readlines()
    bboxes = [list(map(float, line.strip().split()[0:])) for line in lines]

    tile_bboxes = [[] for _ in range(num_tiles)]

    # Ceil needed due to img_width being the original and not the padded image width
    nr_tiles_width = math.ceil(img_width / tile_size)

    for bbox in bboxes:
        label, cx, cy, bw, bh = bbox
        intersect_areas = []
        for i in range(num_tiles):
            row, col = divmod(i, nr_tiles_width)
            x_start, y_start = col * tile_size, row * tile_size
            x_end, y_end = x_start + tile_size, y_start + tile_size

            xA = cx * img_width - bw * img_width / 2
            yA = cy * img_height - bh * img_height / 2
            xB = cx * img_width + bw * img_width / 2
            yB = cy * img_height + bh * img_height / 2

            original_area = (xB - xA) * (yB - yA)

            # Compute intersection area
            xA = max(xA, x_start)
            yA = max(yA, y_start)
            xB = min(xB, x_end)
            yB = min(yB, y_end)

            if xA < xB and yA < yB:
                intersect_area = (xB - xA) * (yB - yA)
                intersect_areas.append((intersect_area, i, original_area))

        for area, tile_idx, original_area in intersect_areas:
            if area > original_area * area_threshold:
                row, col = divmod(tile_idx, nr_tiles_width)
                x_offset, y_offset = col * tile_size, row * tile_size
                new_cx = (cx * img_width - x_offset) / tile_size
                new_cy = (cy * img_height - y_offset) / tile_size
                new_cx = min(1, max(0, new_cx))
                new_cy = min(1, max(0, new_cy))
                # Convert bw and bh to absolute pixel values based on the original image dimensions
                absolute_bw = bw * img_width
                absolute_bh = bh * img_height

                # Now, get the new relative dimensions for the bounding box within the tile
                new_bw = absolute_bw / tile_size
                new_bh = absolute_bh / tile_size
                new_bw = min(1, max(0, new_bw))
                new_bh = min(1, max(0, new_bh))

                tile_bboxes[tile_idx].append([label, new_cx, new_cy, new_bw, new_bh])

    return tile_bboxes


def main(
    original_data_root, tiled_data_root, tile_size, threshold_nonzero_pixels
):
    # img_dirs = ["images/train", "images/val"]
    img_dirs = ["images/val", "images/train"]
    label_dirs = ["labels/val", "labels/train"]

    tiles_info = []
    no_filtered_tiles = 0
    # Loops input dirs
    for img_dir, label_dir in zip(img_dirs, label_dirs):
        orig_img_path = os.path.join(original_data_root, img_dir)
        orig_label_path = os.path.join(original_data_root, label_dir)

        tiled_img_path = os.path.join(tiled_data_root, img_dir)
        tiled_label_path = os.path.join(tiled_data_root, label_dir)

        os.makedirs(tiled_img_path, exist_ok=True)
        os.makedirs(tiled_label_path, exist_ok=True)

        # Loops images
        for img_name in tqdm(os.listdir(orig_img_path)):
            img = cv2.imread(os.path.join(orig_img_path, img_name))
            assert img is not None, f"Failed to read image {img_name}"
            padded_img = pad_image(img, tile_size)

            tiles = split_image(padded_img, tile_size)

            base_name = os.path.splitext(img_name)[0]
            tile_names = []
            label_name = base_name + ".txt"

            tile_bboxes = split_labels(
                os.path.join(orig_label_path, label_name),
                tile_size,
                len(tiles),
                img.shape[1],
                img.shape[0],
                0.2,
            )
            # Loops created tiles
            for idx, tile in enumerate(tiles):
                tile_name = f"{base_name}_{idx}.png"
                no_bboxes = len(tile_bboxes[idx])
                zero_count, non_zero_pixels = count_intensity_pixels(tile)
                if no_bboxes > 0 or non_zero_pixels >= threshold_nonzero_pixels:
                    tile_names.append(tile_name)
                    with open(
                        os.path.join(tiled_label_path, f"{base_name}_{idx}.txt"), "w"
                    ) as f:
                        cv2.imwrite(os.path.join(tiled_img_path, tile_name), tile)
                        for bbox in tile_bboxes[idx]:
                            f.write(f"{' '.join(map(str, bbox))}\n")
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
    parser.add_argument("--tile_size", type=int, required=True)
    parser.add_argument("--threshold_zero_dim", required=False, type=int, default=None)

    args = parser.parse_args()

    orig_data = os.path.join(dataset_root, args.dataset)
    tile_size = args.tile_size
    threshold_zero_dim = args.threshold_zero_dim

    tiled_data_root = f"{dataset_root}/{args.dataset}_tiled_filtered_{tile_size}x{tile_size}"
    main(
        orig_data,
        tiled_data_root,
        tile_size,
        threshold_nonzero_pixels=threshold_zero_dim * threshold_zero_dim,
    )
