import os
from pathlib import Path
import re

def extract_substring(filename):
    match = re.search("(img\d+)\.tif", filename)
    if match:
        return match.group(1)
    print(f"Found filename with non-matching pattern: {filename}")


def rename(in_dir):
    count_unlabeled = 0
    images_dir = Path(os.path.join(in_dir, "images", "train"))
    labels_dir = Path(os.path.join(in_dir, "labels", "train"))
    for imag_name in list(images_dir.iterdir()):
        img_no = extract_substring(imag_name.__str__())
        if "Vegas" in imag_name.__str__():
            prefix = "SN3_roads_train_AOI_2_Vegas_PS-MS_"
            prefix2 = "vegas"
        elif "Paris" in imag_name.__str__():
            prefix = "SN3_roads_train_AOI_3_Paris_PS-MS_"
            prefix2 = "paris"
        elif "Shanghai" in imag_name.__str__():
            prefix = "SN3_roads_train_AOI_4_Shanghai_PS-MS_"
            prefix2 = "shanghai"
        elif "Khartoum" in imag_name.__str__():
            prefix = "SN3_roads_train_AOI_5_Khartoum_PS-MS_"
            prefix2 = "khartoum"
        else:
            raise ValueError(imag_name)
        os.rename(imag_name, f"{imag_name.parent}/{prefix2}{img_no}.tif")
        labels_name = labels_dir / f"{prefix}{img_no}.txt"
        if labels_name.is_file():
            os.rename(labels_name, f"{labels_name.parent}/{prefix2}{img_no}.txt")
        else:
            count_unlabeled += 1
    print(count_unlabeled)


rename("/media/reza/f49c2356-9d18-4385-b433-803e3d56e4991/datasets/SpaceNet-3-OD")