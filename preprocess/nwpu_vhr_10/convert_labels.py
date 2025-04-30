import re
from PIL import Image
from pathlib import Path

from misc.util import mkdir

dir_imgs = Path("/home/alireza/resources/datasets/NWPU-VHR-10/orig/positives")
dir_labels = Path("/home/alireza/resources/datasets/NWPU-VHR-10/labels_orig")
dir_labels_processed = Path(
    "/home/alireza/resources/datasets/NWPU-VHR-10/labels_ultralytics"
)
mkdir(dir_labels_processed.__str__())


def convert_to_coco(x1, y1, x2, y2, img_width, img_height):
    # Convert (x1, y1, x2, y2) box format to (x_center, y_center, w, h) box format
    x_center = ((x2 + x1) / 2.0) / img_width
    y_center = ((y2 + y1) / 2.0) / img_height
    center_w = (x2 - x1) / img_width
    center_h = (y2 - y1) / img_height

    return [x_center, y_center, center_w, center_h]


def extract_coords(string):
    pattern = r"\(\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?)\s*\),\s*\(\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?)\s*\),\s*(\d+)"
    match = re.search(pattern, string)
    if match:
        x1, _, y1, _, x2, _, y2, _, int_val = match.groups()
        return float(x1), float(y1), float(x2), float(y2), int(int_val) - 1
    else:
        raise ValueError("Invalid format")


for img_labels in dir_labels.iterdir():
    path_img = dir_imgs / f"{img_labels.stem}.jpg"
    img = Image.open(path_img)
    i_width = img.width
    i_height = img.height
    with open(img_labels, "r") as f_orig_labels, open(
        dir_labels_processed / f"{img_labels.stem}.txt", "w"
    ) as f_new_labels:
        # iterate through all lines in f_orig_labels
        for line in f_orig_labels:
            if line == "\n":
                print("Line is empty")
                continue
            (
                x_top_left,
                y_top_left,
                x_bottom_right,
                y_bottom_right,
                class_label,
            ) = extract_coords(line)
            x_c, y_c, cw, ch = convert_to_coco(
                x_top_left,
                y_top_left,
                x_bottom_right,
                y_bottom_right,
                i_width * 1.0,
                i_height * 1.0,
            )
            assert x_c <= 1.0 and y_c <= 1.0 and cw <= 1.0 and ch <= 1.0
            f_new_labels.write(f"{class_label} {x_c} {y_c} {cw} {ch}\n")
