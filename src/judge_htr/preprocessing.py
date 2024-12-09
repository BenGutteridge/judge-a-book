"""Utility functions for preprocessing data in 00_iam_preprocessing.py"""

import os
from collections import defaultdict
from loguru import logger
from pathlib import Path
import pandas as pd
from PIL import Image

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./htr-icl-googlevision-key.json"


def combine_gt_txt_files(input_dir: Path, output_dir: Path):
    """Turn stacks of txt files that split up the transcript by line into a single txt file."""
    file_groups = defaultdict(list)
    count = 0

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            # Extract the prefix (ddd_ddd_ddd)
            prefix = "_".join(filename.split("_")[:3])

            with open(input_dir / filename, "r") as file:
                content = file.read()
                file_groups[prefix].append(content)

    # Combine the content of each group and write to a new file
    for prefix, contents in file_groups.items():
        combined_filename = f"{prefix}.txt"
        with open(output_dir / combined_filename, "w") as combined_file:
            combined_file.write("".join(contents))
        logger.info(f"Written {combined_filename}")
        count += 1

    logger.info(f"{count} files combined successfully.")


def txt2str(filepath: Path) -> str:
    """Convert txt file to string"""
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    content.replace("=\n : ", "-\n")  # TODO check!
    return content


# Not yet used
def compress_image(
    image_path: Path,
    output_folder: Path,
    size_threshold_kb: int,
    quality_step: int = 5,
) -> Path:
    """
    Compresses a JPG image to fit under a specific size threshold in KB and saves it to a new folder.

    Parameters:
    image_path (str): Path to the original image file.
    output_folder (Path): Path to the folder where the compressed image will be saved.
    size_threshold_kb (int): Target file size threshold in KB.
    quality_step (int): Step size to decrease quality (default 5).

    Returns:
    Path to compressed image (Path)
    """
    # Ensure the output folder exists
    output_folder.mkdir(exist_ok=True)

    # Open the image
    img = Image.open(image_path)

    # Set the initial quality and output path
    quality = 95  # Start with high quality
    compressed_image_path = (
        output_folder / f"{image_path.stem}_compressed{image_path.suffix}"
    )

    # Won't repeat if compressed already
    if compressed_image_path.exists():
        if (
            (compressed_image_path.stat().st_size / 1024) <= size_threshold_kb
        ) or quality <= 10:
            return compressed_image_path

    # Compress and save the image, adjusting the quality until it's below the threshold
    while True:
        # Save image with current quality
        img.save(compressed_image_path, "JPEG", quality=quality)

        # If the file size is under the threshold, stop compressing
        file_size_kb = compressed_image_path.stat().st_size / 1024
        if file_size_kb <= size_threshold_kb or quality <= 10:
            break

        # Reduce the quality for the next iteration
        quality -= quality_step

    return compressed_image_path


def get_handwritten_vertical_bounds(
    xml_path: Path, sf_upper: int = 30, sf_lower: int = 80
) -> tuple[int, int]:
    """
    Parse XML to get vertical bounds (top and bottom) of the handwritten part of an IAM image.

    Parameters:
    - xml_path (Path): Path to the XML file.
    - sf (int): Safety factor to add to the top and bottom bounds (in pixels).

    Returns:
    - Tuple[int, int]: (top, bottom) vertical bounds for cropping.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the first `asy` value for the top bound
    first_line = root.find(".//handwritten-part/line")
    top = int(first_line.get("asy")) if first_line is not None else 0

    # Find the last `y` value in `<cmp>` elements inside `<word>` tags for the bottom bound
    last_y = 0
    for word in root.findall(".//handwritten-part/line/word/cmp"):
        y = int(word.get("y"))
        last_y = max(last_y, y)

    bottom = last_y

    # Add safety factor

    return top - sf_upper, bottom + sf_lower


def crop_image_to_handwritten_part(
    image_path: Path,
    xml_path: Path,
    output_path: Path,
) -> None:
    """
    Crop an image to just the handwritten part based on XML data.

    Parameters:
    - image_path (Path): Path to the image file.
    - xml_path (Path): Path to the XML file.
    - output_path (Path): Path to save the cropped image.
    """

    if output_path.exists():
        logger.info(f"Output path {output_path} already exists.")
        return

    # Get vertical bounds from XML
    top, bottom = get_handwritten_vertical_bounds(xml_path)

    if bottom - top < 300:
        logger.error(f"Vertical bounds too narrow: {top=}, {bottom=}")

    # Open the image
    with Image.open(image_path) as img:
        # Get image dimensions and set horizontal bounds to full width
        img_width, img_height = img.size
        left, right = 0, img_width

        # Ensure bottom bound is within image height
        bottom = min(bottom, img_height)

        # Crop the image to the bounding box
        cropped_img = img.crop((left, top, right, bottom))
        # Save the cropped image
        cropped_img.save(output_path)
        logger.info(f"Cropped image saved to {output_path}")
