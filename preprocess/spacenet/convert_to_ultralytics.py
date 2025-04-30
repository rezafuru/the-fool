import json
import os
from collections import defaultdict

import rasterio
from shapely import geometry
from math import floor
import re
global unlabeled_count

unlabeled_count = 0

class_frequencies = defaultdict(int)
def extract_substring(filename):
    match = re.search("(img\d+)\.tif", filename)
    if match:
        return match.group(1)
    print(f"Found filename with non-matching pattern: {filename}")


def darknet_bb_from_feature(feature, affine_transform):
    global class_frequencies
    """Build a Darknet style bounding box from a geojson style feature."""
    poly = geometry.shape(feature["geometry"])
    obj_class = feature["properties"]["road_type"]
    obj_class = int(obj_class) - 1
    assert obj_class >= 0
    center = poly.centroid
    # Apply inverse affine transform to get image coordinates
    x_center, y_center = ~affine_transform * (center.x, center.y)
    bbox = poly.bounds
    # Calculate width and height in image coordinates
    width = (bbox[2] - bbox[0]) / affine_transform[0]
    height = (bbox[3] - bbox[1]) / -affine_transform[4]
    out = (str(obj_class), x_center, y_center, width, height)
    class_frequencies[obj_class] += 1
    return out


def convert_to_ultralytics(bbox, img_width, img_height):
    """Convert bounding box from Darknet to Ultralytics format."""
    obj_class, x_center, y_center, width, height = bbox
    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    return (obj_class, x_center, y_center, width, height)


def process_dataset(root_dir):
    global unlabeled_count
    image_dir = os.path.join(root_dir, "images8bit")
    geojson_dir = os.path.join(root_dir, "geojson")
    label_dir = os.path.join(root_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(".tif"):
            base_name = os.path.splitext(image_filename)[0]
            img_no = extract_substring(image_filename)
            if "Vegas" in image_filename:
                geojson_filename = (
                    f"SN3_roads_train_AOI_2_Vegas_geojson_roads_{img_no}.geojson"
                )
            elif "Paris" in image_filename:
                geojson_filename = (
                    f"SN3_roads_train_AOI_3_Paris_geojson_roads_{img_no}.geojson"
                )
            elif "Shanghai" in image_filename:
                geojson_filename = (
                    f"SN3_roads_train_AOI_4_Shanghai_geojson_roads_{img_no}.geojson"
                )
            elif "Khartoum" in image_filename:
                geojson_filename = (
                    f"SN3_roads_train_AOI_5_Khartoum_geojson_roads_{img_no}.geojson"
                )
            else:
                raise ValueError()
            geojson_path = os.path.join(geojson_dir, geojson_filename)
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            raster_file = os.path.join(image_dir, image_filename)

            # Get affine transform from raster file
            with rasterio.open(raster_file) as src:
                aff = src.transform
                img_width, img_height = src.width, src.height

            # Load GeoJSON file
            if os.path.isfile(geojson_path):
                with open(geojson_path) as src:
                    fc = json.load(src)
            else:
                print(f"{geojson_path} not found")
                unlabeled_count += 1
                continue

            # Convert features to Darknet format and then to Ultralytics format
            with open(label_path, "w") as file:
                for feature in fc["features"]:
                    darknet_bbox = darknet_bb_from_feature(feature, aff)
                    ultralytics_bbox = convert_to_ultralytics(
                        darknet_bbox, img_width, img_height
                    )
                    file.write(" ".join(map(str, ultralytics_bbox)) + "\n")


# Example usage
root_dir = os.path.expanduser("~/resources/datasets/SpaceNet-3")
process_dataset(root_dir)
print(unlabeled_count)

print(class_frequencies)