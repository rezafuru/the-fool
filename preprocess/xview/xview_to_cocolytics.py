import json
import os
from pathlib import Path

from PIL import Image

sorted_class_names = {
    "Fixed-Wing Aircraft": 11,
    "Small aircraft": 12,
    "Passenger/Cargo Plane": 13,
    "Helicopter": 15,
    "Passenger Vehicle": 17,
    "Small Car": 18,
    "Bus": 19,
    "Pickup Truck": 20,
    "Utility Truck": 21,
    "Truck": 23,
    "Cargo Truck": 24,
    "Truck Tractor w/ Box Trailer": 25,
    "Truck Tractor": 26,
    "Trailer": 27,
    "Truck Tractor w/ Flatbed Trailer": 28,
    "Truck Tractor w/ Liquid Tank": 29,
    "Crane Truck": 32,
    "Railway Vehicle": 33,
    "Passenger Car": 34,
    "Cargo/Container Car": 35,
    "Flat Car": 36,
    "Tank Car": 37,
    "Locomotive": 38,
    "Maritime Vessel": 40,
    "Motorboat": 41,
    "Sailboat": 42,
    "Tugboat": 44,
    "Barge": 45,
    "Fishing Vessel": 47,
    "Ferry": 49,
    "Yacht": 50,
    "Container Ship": 51,
    "Oil Tanker": 52,
    "Engineering Vehicle": 53,
    "Tower Crane": 54,
    "Container Crane": 55,
    "Reach Stacker": 56,
    "Straddle Carrier": 57,
    "Mobile Crane": 59,
    "Dump Truck": 60,
    "Haul Truck": 61,
    "Tractor": 62,
    "Front Loader/Bulldozer": 63,
    "Excavator": 64,
    "Cement Mixer": 65,
    "Ground Grader": 66,
    "Hut/Tent": 71,
    "Shed": 72,
    "Building": 73,
    "Aircraft Hangar": 74,
    "Damaged/Demolished Building": 76,
    "Facility": 77,
    "Construction Site": 79,
    "Vehicle Lot": 83,
    "Helipad": 84,
    "Storage Tank": 85,
    "Storage tank": 86,
    "Shipping Container Lot": 89,
    "Shipping Container": 91,
    "Pylon": 93,
    "Tower": 94,
    "Unknown Label 1": 75,
    "Unknown Label 2": 82,

}
class_lookup = {
    11: 0,
    12: 1,
    13: 2,
    15: 3,
    17: 4,
    18: 5,
    19: 6,
    20: 7,
    21: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 15,
    32: 16,
    33: 17,
    34: 18,
    35: 19,
    36: 20,
    37: 21,
    38: 22,
    40: 23,
    41: 24,
    42: 25,
    44: 26,
    45: 27,
    47: 28,
    49: 29,
    50: 30,
    51: 31,
    52: 32,
    53: 33,
    54: 34,
    55: 35,
    56: 36,
    57: 37,
    59: 38,
    60: 39,
    61: 40,
    62: 41,
    63: 42,
    64: 43,
    65: 44,
    66: 45,
    71: 46,
    72: 47,
    73: 48,
    74: 49,
    76: 50,
    77: 51,
    79: 52,
    83: 53,
    84: 54,
    85: 55,
    86: 56,
    89: 57,
    91: 58,
    93: 59,
    94: 60,
    75: 61,
    82: 62,
}

MISSING_IMGS = ["1395.tif", "18.tif", "1831.tif", "88.tif", "1469.tif"]

def geojson_to_coco_ultralytics(geojson_path, img_folders, label_folders):
    # Load the geojson file
    with open(geojson_path, "r") as f:
        data = json.load(f)
    for img_folder, label_folder in zip(img_folders, label_folders):

        # Ensure destination directory exists
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        # Extract features from geojson data
        features = data["features"]

        # Get a list of all unique image IDs
        all_img_ids = {feature["properties"]["image_id"] for feature in features}
        for missing_id in MISSING_IMGS:
            all_img_ids.remove(missing_id)

        # Iterate over features and extract necessary data
        for img_id in all_img_ids:
            img_path = os.path.join(img_folder, img_id)

            # Open the corresponding image and get its size
            # with Image.open(img_path) as img:
            #     width, height = img.size

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                print(f"Not found {img_path}")
                continue

            # Filter features for the current image
            img_features = [f for f in features if f["properties"]["image_id"] == img_id]

            # Write to the output file
            output_txt_path = os.path.join(label_folder, os.path.splitext(img_id)[0] + ".txt")
            with open(output_txt_path, "a") as out_file:
                for feature in img_features:
                    properties = feature["properties"]
                    bounds = properties["bounds_imcoords"]
                    xmin, ymin, xmax, ymax = list(map(int, bounds.split(",")))

                    # Convert to COCO Ultralytics format
                    x_center = ((xmin + xmax) / 2) / width
                    y_center = ((ymin + ymax) / 2) / height
                    b_width = (xmax - xmin) / width
                    b_height = (ymax - ymin) / height

                    # Construct the formatted string
                    formatted_str = f"{class_lookup[int(properties['type_id'])]} {x_center} {y_center} {b_width} {b_height}\n"
                    out_file.write(formatted_str)


if __name__ == "__main__":
    xview_root = Path(os.path.expanduser("~/resources/datasets/xView-flat/"))
    geojson_path = xview_root / "xView_train.geojson"
    img_train_folder = xview_root / "images" / "train"
    img_val_folder = xview_root / "images" / "val"
    label_train_folder = xview_root / "labels" / "train"
    label_val_folder = xview_root / "labels" / "val"

    img_folders = [img_train_folder]
    label_folders = [label_train_folder]

    geojson_to_coco_ultralytics(geojson_path, img_folders, label_folders)
