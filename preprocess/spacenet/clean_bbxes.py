import os


"""
removes nested boxes
"""

def is_nested(box1, box2):
    # Check if box1 is completely inside box2
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert from center, width, height to min and max coordinates
    min_x1, min_y1, max_x1, max_y1 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    min_x2, min_y2, max_x2, max_y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    # Check if box1 is inside box2
    return min_x2 < min_x1 and min_y2 < min_y1 and max_x2 > max_x1 and max_y2 > max_y1


def clean_labels(root_dir):
    for subset in ["train", "val"]:
        label_dir = os.path.join(root_dir, "labels", subset)
        clean_label_dir = os.path.join(root_dir, "labels_clean", subset)
        os.makedirs(clean_label_dir, exist_ok=True)  # Ensure the clean directory exists

        for label_file in os.listdir(label_dir):
            input_file_path = os.path.join(label_dir, label_file)
            output_file_path = os.path.join(clean_label_dir, label_file)

            with open(input_file_path, "r") as f:
                lines = f.readlines()

            boxes = [
                list(map(float, line.split()[1:])) for line in lines
            ]  # Extract bbox info
            valid_boxes = []
            for i, box in enumerate(boxes):
                if not any(
                    is_nested(box, other_box)
                    for j, other_box in enumerate(boxes)
                    if i != j
                ):
                    valid_boxes.append(lines[i])

            with open(output_file_path, "w") as f:
                f.writelines(valid_boxes)


# Usage
root_dir = os.path.expanduser("~/resources/datasets/SpaceNet-3-OD-custom-split")  # Replace with your root directory path
clean_labels(root_dir)
