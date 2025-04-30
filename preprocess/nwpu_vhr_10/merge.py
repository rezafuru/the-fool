import shutil
from distutils.dir_util import copy_tree
from pathlib import Path

root_orig = Path("/home/alireza/resources/datasets/NWPU-VHR-10/orig")
root_merged = Path("/home/alireza/resources/datasets/NWPU-VHR-10/merged")

# copy directory of images root_orig/positives to root_merged/dataset
# iterate through each image in root_orig, copy it and save it with the name {index}.jpg,
# where index starts at the number of samples in root_orig/positives
copy_tree(
    "/home/alireza/resources/datasets/NWPU-VHR-10/orig/positives",
    "/home/alireza/resources/datasets/NWPU-VHR-10/merged/dataset",
)
dir_negatives = root_orig / "negatives"
dataset_merged = root_merged / "dataset"
img_paths = list(dir_negatives.iterdir())
img_paths.sort()
idx = len(list((root_orig / "positives").iterdir()))
for img in img_paths:
    idx += 1
    # copy image in path img to dataset merged with name idx.jpg
    img_name = f"{idx}.jpg"
    img_dst = dataset_merged / img_name
    shutil.copy(src=img.__str__(), dst=img_dst.__str__())
