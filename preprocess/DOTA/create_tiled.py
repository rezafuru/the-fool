import math
import os
import cv2
import numpy as np


def pad_image(img, tile_size):
    h, w = img.shape[:2]
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)

def upsample_image(img, tile_size, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    new_h = ((h + tile_size - 1) // tile_size) * tile_size
    new_w = ((w + tile_size - 1) // tile_size) * tile_size
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)



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


def main(original_data_root, tiled_data_root, tile_size):
    # img_dirs = ["images/train", "images/val"]
    img_dirs = ["images/train", "images/val"]
    label_dirs = ["labels/train", "labels/val"]

    tiles_info = []

    # Loops input dirs
    for img_dir, label_dir in zip(img_dirs, label_dirs):
        orig_img_path = os.path.join(original_data_root, img_dir)
        orig_label_path = os.path.join(original_data_root, label_dir)

        tiled_img_path = os.path.join(tiled_data_root, img_dir)
        tiled_label_path = os.path.join(tiled_data_root, label_dir)

        os.makedirs(tiled_img_path, exist_ok=True)
        os.makedirs(tiled_label_path, exist_ok=True)

        # Loops images
        for img_name in os.listdir(orig_img_path):
            img = cv2.imread(os.path.join(orig_img_path, img_name))
            assert img is not None, f"Failed to read image {img_name}"

            padded_img = upsample_image(img, tile_size)

            tiles = split_image(padded_img, tile_size)

            base_name = os.path.splitext(img_name)[0]
            tile_names = []

            # Loops created tiles
            for idx, tile in enumerate(tiles):
                tile_name = f"{base_name}_{idx}.png"
                tile_names.append(tile_name)
                cv2.imwrite(os.path.join(tiled_img_path, tile_name), tile)

            tiles_info.append(" ".join(tile_names))

            label_name = base_name + ".txt"

            tile_bboxes = split_labels(
                os.path.join(orig_label_path, label_name),
                tile_size,
                len(tiles),
                img.shape[1],
                img.shape[0],
                0.2,
            )

            for idx, bboxes in enumerate(tile_bboxes):
                with open(
                    os.path.join(tiled_label_path, f"{base_name}_{idx}.txt"), "w"
                ) as f:
                    for bbox in bboxes:
                        f.write(f"{' '.join(map(str, bbox))}\n")

    with open(os.path.join(tiled_data_root, "tiles_info.txt"), "w") as f:
        f.write("\n".join(tiles_info))


if __name__ == "__main__":
    tile_size = 512
    dataset_name = "DOTA-2"
    datasets_root = os.path.expanduser("~/resources/datasets")
    original_data_root = f"{datasets_root}/{dataset_name}"
    tiled_data_root = f"{datasets_root}/{dataset_name}_tiled_{tile_size}x{tile_size}_upsampled"
    main(original_data_root, tiled_data_root, tile_size)
