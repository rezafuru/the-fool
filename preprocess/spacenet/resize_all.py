import os
from pathlib import Path

import PIL.Image
from PIL import Image
import tifffile as tif

def resize_all(in_dir, out_dir, size):
    out_dir = Path(out_dir)
    for img_file in list(Path(in_dir).iterdir()):
        assert "tif" in img_file.__str__()
        img_arr = tif.imread(img_file)
        img = Image.fromarray(img_arr).resize(size=(size, size), resample=PIL.Image.BICUBIC)
        dest = out_dir / out_dir / f"{Path(img_file).stem}.tif"
        img.save(dest)



in_dir= "/media/reza/f49c2356-9d18-4385-b433-803e3d56e4991/datasets/SpaceNet-3/images8bit"
out_dir= "/media/reza/f49c2356-9d18-4385-b433-803e3d56e4991/datasets/SpaceNet-3/images8bit_resized"

resize_all(in_dir, out_dir, 1536)