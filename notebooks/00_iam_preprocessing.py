# %% [markdown]
# # Convert IAM dataset into pandas dataframes and save for downstream experiment notebooks.

# %% [markdown]
# ### Setup
from judge_htr import data
from loguru import logger
from judge_htr.preprocessing import crop_image_to_handwritten_part
from judge_htr.ocr_wrappers import (
    googleocr_img2txt,
    azure_img2txt,
    textract_img2txt,
)
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import xml.etree.ElementTree as ET
import pickle
from matplotlib import pyplot as plt

load_dotenv()
tqdm.pandas()

data_dir = data / "IAM Handwriting DB"

ocr_engines = [
    "google_ocr",
    "azure",
    "textract",
]

# %% [markdown]
# Get per-page information about author, etc


def load_forms_to_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Load the forms.txt file into a pandas DataFrame.

    Args:
    - file_path (Path): The path to the forms.txt file.

    Returns:
    - pd.DataFrame: A DataFrame containing the parsed form data.
    """
    columns = [
        "id",
        "writer_id",
        "num_sentences",
        "segmentation",
        "total_lines",
        "segmented_lines",
        "total_words",
        "segmented_words",
    ]

    data = []

    with file_path.open("r") as file:
        for line in file:
            # Ignore lines starting with non-data (e.g., comments, empty lines)
            if line.startswith("#") or not line.strip():
                continue

            # Split the line into parts based on the format
            parts = line.strip().split()

            # Append parsed data as a row in the data list
            data.append(
                [
                    parts[0],  # id
                    int(parts[1]),  # writer_id
                    int(parts[2]),  # num_sentences
                    parts[3],  # segmentation
                    int(parts[4]),  # total_lines
                    int(parts[5]),  # segmented_lines
                    int(parts[6]),  # total_words
                    int(parts[7]),  # segmented_words
                ]
            )

    # Create the DataFrame with the specified columns
    df = pd.DataFrame(data, columns=columns)
    return df


file_path = data_dir / "ascii/forms.txt"
logger.info("Loading form metadata into DataFrame...")
df = load_forms_to_dataframe(file_path)

# %% [markdown]
# Get ground truth text strings


def extract_machine_printed_text(file_path: Path) -> str:
    """
    Extracts text from <machine-print-line> tags within <machine-printed-part> in an XML file.

    Args:
    - file_path (Path): The path to the .xml file.

    Returns:
    - str: The extracted text with lines separated by newlines.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the machine-printed-part section
    machine_printed_part = root.find("machine-printed-part")

    # Extract text from each machine-print-line within the machine-printed-part
    lines = [
        line.get("text") for line in machine_printed_part.findall("machine-print-line")
    ]

    # Join lines with newlines
    return "\n".join(lines)


def extract_handwritten_text(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <handwritten-part> section
    handwritten_part = root.find("handwritten-part")

    # Extract text from each <line> within <handwritten-part>
    text_lines = []
    for line in handwritten_part.findall("line"):
        # Get the text attribute of each line and add it to the list
        line_text = line.get("text")
        if line_text:
            text_lines.append(line_text)

    # Join lines with newline characters
    extracted_text = "\n".join(text_lines)

    # Replace HTML entities with their corresponding characters
    extracted_text = extracted_text.replace("&quot;", '"')
    extracted_text = extracted_text.replace("&amp;", "&")

    return extracted_text


logger.info("Extracting ground-truth text from XML files...")
df["gt"] = df["id"].progress_apply(
    lambda x: extract_handwritten_text(data_dir / f"xml/{x}.xml")
)


# %% [markdown]
# Get IAM document images cropped to contain handwriting only
# (they also include the machine-printed text, which we need to remove to make it a HTR task)

xml_dir = data_dir / "xml"
img_dirs = [data_dir / s for s in ["formsA-D", "formsE-H", "formsI-Z"]]
output_dir = data_dir / "forms_handwriting_only"
output_dir.mkdir(exist_ok=True)

logger.info("Cropping images to handwritten parts only...")
for img_dir in img_dirs:
    for img_path in tqdm(img_dir.glob("*.png"), total=len(list(img_dir.glob("*.png")))):
        xml_path = xml_dir / f"{img_path.stem}.xml"
        output_path = output_dir / f"{img_path.stem}.png"
        crop_image_to_handwritten_part(img_path, xml_path, output_path)

df["img_path"] = df["id"].apply(lambda x: data_dir / f"forms_handwriting_only/{x}.png")


# %% [markdown]
# Get OCR transcriptions from images
ocr_outputs = {engine: {} for engine in ocr_engines}

ocr_to_text_funcs = {
    "google_ocr": googleocr_img2txt,
    "azure": azure_img2txt,
    "textract": textract_img2txt,
}

# Looping allows recovery if interrupted
for engine in ocr_engines:
    logger.info(f"\nRunning {engine} OCR on images...")
    saved_path = data_dir / f"ocr_{engine}.pkl"
    if saved_path.exists():
        with open(saved_path, "rb") as f:
            ocr_outputs[engine] = pickle.load(f)
        continue
    for i, row in tqdm(df.iterrows(), total=len(df)):
        ocr_outputs[engine][row["id"]] = ocr_to_text_funcs[engine](row["img_path"])
    # Pickle the dict just in case
    with open(data_dir / f"{engine}.pkl", "wb") as f:
        pickle.dump(ocr_outputs[engine], f)

# Add dfs to main df
for engine in ocr_engines:
    df[engine] = df["id"].apply(lambda x: ocr_outputs[engine][x])

# %% [markdown]
# Save the df
df_filepath = data_dir / "iam.pkl"
df.to_pickle(df_filepath)
logger.info(f"\nSaved df to {df_filepath}")

# %% [markdown]
# # Multi-page mode
# We have per-page OCR transcriptions, but we want to work with multi-page documents. So lets construct some.
#
# Quick check: how many multipage docs can we make that have the same writer_id? (i.e. same handwriting)
plt.bar(
    *zip(
        *{
            x: ([x for x in df[["writer_id"]].value_counts().tolist() if x > 1]).count(
                x
            )
            for x in set(
                [x for x in df[["writer_id"]].value_counts().tolist() if x > 1]
            )
        }.items()
    )
)
plt.xlabel("Number of pages per writer")
plt.ylabel("Number of writers")
# plt.show()

# %% [markdown]
# Convert into multi-page docs by combining pages *from the same writer*
#
# Let's use 2-3 pages per writer (2 by default, with a 3 if there's an odd number)
# with the option of using a higher minimum page count.
#
# Output used for downstream experiments and plot/table notebooks.

dfmp = (
    df.groupby("writer_id")[["gt", "img_path"] + ocr_engines]
    .agg(lambda x: list(x))
    .reset_index()
)

min_page_count = 10  # min_page_count=2 used for most experiments; the others are for 03_pagecount_plot.py

for min_page_count in range(2, 11):
    logger.info(f"\nFinding docs with minimum {min_page_count} pages...")
    df = dfmp[
        dfmp["gt"].apply(len) >= min_page_count
    ]  # ignore single-page writers/writers with insufficient numbers of pages

    new_rows = []
    for i, row in df.iterrows():
        n = len(row["gt"])
        idxs = [min_page_count * i for i in range(n // min_page_count + 1)]
        remainder = n % min_page_count
        if remainder != 0:
            idxs[-1] += remainder
        for i, j in zip(idxs[:-1], idxs[1:]):
            new_row = row.copy()
            new_row["gt"] = row["gt"][i:j]
            new_row["img_path"] = row["img_path"][i:j]
            for engine in ocr_engines:
                new_row[engine] = row[engine][i:j]
            new_rows.append(new_row)

    df_multipage = pd.DataFrame(new_rows).reset_index(drop=True)

    logger.info(
        f"\nMultipage doc counts (minimum {min_page_count} pages):\n"
        f"{df_multipage['gt'].apply(len).value_counts()}"
    )

    df_filepath = data_dir / f"iam_multipage_minpages={min_page_count:02d}.pkl"
    df_multipage.to_pickle(df_filepath)
    logger.info(
        f"\nSaved multipage df to {df_filepath} with minimum page count of {min_page_count}"
    )


# %% [markdown]
# Make a dataset of documents of 2-10 pages, with 10 documents per page count
#
# Used downstream in 03_pagecount_plot.py

seed = 0
dfs = []
for page_count in range(2, 11):
    df_filepath = data_dir / f"iam_multipage_minpages={page_count:02d}.pkl"
    df = pd.read_pickle(df_filepath)
    # Filter to only docs with precisely `page_count` pages
    df = df[df["gt"].apply(len) == page_count]
    max_samples = len(df)
    # Sample 10 docs per length, or the maximum number of docs available
    dfs.append(df.sample(min(10, max_samples), random_state=seed))

# %%
df = pd.concat(dfs).reset_index(drop=True)

df.to_pickle(data_dir / "iam_multipage_2-10_pages_10_docs.pkl")

# %% [markdown]
