import dota_utils as util
import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 93312000000
classes = [
    "building",
    "prefabricated-house",
    "cable-tower",
    "quarry",
    "landslide",
    "pool",
    "well",
    "ship",
    "cultivation-mesh-cage",
    "vehicle",
]
"""
classes = ['plane',
        'passenger',
        'building',
        'truck',
        'railway',
        'maritime',
        'engineering',
        'else']
"""


# trans dota format to format YOLO(darknet) required
def dota2darknet(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
    """
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    filelist = util.GetFileFromThisRootDir(txtpath)
    filelist.sort()
    for fullname in filelist:
        objects = util.parse_dota_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + ".png")
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + ".txt"), "w") as f_out:
            for obj in objects:
                poly = obj["poly"]
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))
                # note: DOTA seems to create files even when bboxes = 0
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                if obj["name"] in extractclassname:
                    id = extractclassname.index(obj["name"])
                else:
                    continue
                outline = str(id) + " " + " ".join(list(map(str, bbox)))
                f_out.write(outline + "\n")


if __name__ == "__main__":
    ## an example

    dota2darknet(
        os.path.expanduser("~/resources/datasets/DOTA-2/images/train"),
        os.path.expanduser("~/resources/datasets/DOTA-2/labels_orig/train"),
        os.path.expanduser("~/resources/datasets/DOTA-2/labels/train"),
        util.CATEGORIES_DOTA_2
    )
