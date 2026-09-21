from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parent

TRAIN_IMAGES = ROOT / "train" / "images"
TRAIN_ANNOTATIONS = ROOT / "train" / "annotations"

VAL_IMAGES = ROOT / "validation" / "images"
VAL_ANNOTATIONS = ROOT / "validation" / "annotations"

OUTPUT = ROOT / "dataset"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def normalize_class(name):
    name = name.strip().lower()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")

    aliases = {
        "crazing": "crazing",
        "inclusion": "inclusion",
        "patch": "patches",
        "patches": "patches",
        "pitted": "pitted_surface",
        "pitted_surface": "pitted_surface",
        "rolled_in": "rolled_in_scale",
        "rolled_in_scale": "rolled_in_scale",
        "scratch": "scratches",
        "scratches": "scratches",
    }

    return aliases.get(name, name)


def find_image(image_folder, xml_file):
    """
    Find image corresponding to XML annotation.
    """

    xml_stem = xml_file.stem

    # First try same filename with common extensions
    for extension in IMAGE_EXTENSIONS:

        candidate = image_folder / f"{xml_stem}{extension}"

        if candidate.exists():
            return candidate

    # Search recursively
    for image in image_folder.rglob("*"):

        if (
            image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
            and image.stem == xml_stem
        ):
            return image

    return None


def convert_xml_to_yolo(xml_file, image_file):

    tree = ET.parse(xml_file)

    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")

    if size is not None:

        width_text = size.findtext("width")
        height_text = size.findtext("height")

        if width_text and height_text:

            width = float(width_text)
            height = float(height_text)

        else:

            with Image.open(image_file) as image:
                width, height = image.size

    else:

        with Image.open(image_file) as image:
            width, height = image.size

    if width <= 0 or height <= 0:

        raise ValueError(
            f"Invalid image dimensions in {xml_file}"
        )

    yolo_lines = []

    for object_element in root.findall("object"):

        class_name = object_element.findtext("name", "")

        class_name = normalize_class(class_name)

        if class_name not in CLASSES:

            print(
                f"[WARNING] Unknown class '{class_name}' "
                f"in {xml_file.name}"
            )

            continue

        bounding_box = object_element.find("bndbox")

        if bounding_box is None:
            continue

        xmin = float(
            bounding_box.findtext("xmin", "0")
        )

        ymin = float(
            bounding_box.findtext("ymin", "0")
        )

        xmax = float(
            bounding_box.findtext("xmax", "0")
        )

        ymax = float(
            bounding_box.findtext("ymax", "0")
        )

        # Keep coordinates inside image
        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        # YOLO format
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height

        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = CLASSES.index(class_name)

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_lines.append(line)

    return yolo_lines


def prepare_split(
    split_name,
    image_folder,
    annotation_folder,
    output_split
):

    print()
    print("=" * 60)
    print(f"Preparing {split_name} dataset")
    print("=" * 60)

    if not image_folder.exists():

        raise FileNotFoundError(
            f"Image folder not found:\n{image_folder}"
        )

    if not annotation_folder.exists():

        raise FileNotFoundError(
            f"Annotation folder not found:\n{annotation_folder}"
        )

    xml_files = sorted(
        annotation_folder.rglob("*.xml")
    )

    if len(xml_files) == 0:

        raise FileNotFoundError(
            f"No XML files found in:\n{annotation_folder}"
        )

    output_images = (
        OUTPUT
        / "images"
        / output_split
    )

    output_labels = (
        OUTPUT
        / "labels"
        / output_split
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    converted = 0

    for xml_file in xml_files:

        image_file = find_image(
            image_folder,
            xml_file
        )

        if image_file is None:

            print(
                f"[WARNING] Image not found for "
                f"{xml_file.name}"
            )

            continue

        try:

            labels = convert_xml_to_yolo(
                xml_file,
                image_file
            )

        except Exception as error:

            print(
                f"[ERROR] {xml_file.name}: {error}"
            )

            continue

        if len(labels) == 0:

            print(
                f"[WARNING] No valid bounding boxes "
                f"in {xml_file.name}"
            )

            continue

        output_image = (
            output_images
            / f"{image_file.stem}.jpg"
        )

        output_label = (
            output_labels
            / f"{image_file.stem}.txt"
        )

        # Convert image to RGB JPEG
        with Image.open(image_file) as image:

            image = image.convert("RGB")

            image.save(
                output_image,
                quality=95
            )

        # Save YOLO labels
        output_label.write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8"
        )

        converted += 1

    print(
        f"{split_name}: {converted} images converted"
    )

    return converted


def create_yaml():

    yaml_file = OUTPUT / "data.yaml"

    yaml_content = f"""
path: {OUTPUT.resolve().as_posix()}

train: images/train

val: images/val

test: images/val

nc: {len(CLASSES)}

names:
"""

    for index, class_name in enumerate(CLASSES):

        yaml_content += (
            f"  {index}: {class_name}\n"
        )

    yaml_file.write_text(
        yaml_content.strip() + "\n",
        encoding="utf-8"
    )

    print()
    print(f"YOLO configuration created:")
    print(yaml_file)


def main():

    print()
    print("=" * 60)
    print("NEU-DET DATASET PREPARATION")
    print("=" * 60)

    train_count = prepare_split(
        "TRAIN",
        TRAIN_IMAGES,
        TRAIN_ANNOTATIONS,
        "train"
    )

    validation_count = prepare_split(
        "VALIDATION",
        VAL_IMAGES,
        VAL_ANNOTATIONS,
        "val"
    )

    create_yaml()

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Training images   : {train_count}"
    )

    print(
        f"Validation images : {validation_count}"
    )

    print(
        f"YOLO dataset      : {OUTPUT}"
    )


if __name__ == "__main__":

    main()