# %% [markdown]
# # Generate scatter plot of all methods with Pareto frontier

from prompts import newpage
from judge_htr import results, data, plots
import pandas as pd
from evaluate import load
from loguru import logger
from tqdm import tqdm
from judge_htr.postprocessing import (
    method_strs,
    gpt_output_postprocessing,
)
import anls_star
from diskcache import Cache
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import os

os.environ["PATH"] += ":/Library/TeX/texbin"  # for latex in figures
from matplotlib import pyplot as plt

import numpy as np


tqdm.pandas()
cache = Cache(data)
cer = load("cer")
plt.rcParams["text.usetex"] = True

tqdm.pandas()

ocr_engines = [
    "azure",
    "google_ocr",
    "textract",
]

# %% [markdown]
# ### Parameters
drop_first_page = False
remove_whitespace = False

if remove_whitespace:
    str_proc = lambda s: gpt_output_postprocessing(s).replace(" ", "")
else:
    str_proc = gpt_output_postprocessing


seed = 0

# IAM (2 pages)
split = 0.5
results_file = f"iam_multipage_minpages=02_split={split:.02f}_seed={seed:02d}"
run_name = "IAM multi-page"
results_path = results / f"{results_file}_checked.pkl"

df = pd.read_pickle(results_path)

# Metric
metric = "CER"  # Character Error Rate
# metric = "ANLS"  # ANLS*

score = {
    "CER": lambda gt, pred: cer.compute(predictions=pred, references=gt),
    "ANLS": lambda gt, pred: anls_star.anls_score(gt=gt, pred=pred),
}[metric]

all_modes = [col.replace("_cost", "") for col in df.columns if col.endswith("_cost")]

# %% [markdown]
# Split up mode into OCR engine, method, and GPT model

df["num_pages"] = df["gt"].apply(len)

mode_tuples = {}
for mode in all_modes:
    if mode in ocr_engines:
        ocr_engine, method, gpt_model = mode, None, None
    elif "->" in mode:
        method, gpt_model = mode.split("->")
        if set(method.split("+")) == set(ocr_engines):
            ocr_engine = None
        elif "+" in method:
            ocr_engine, method = method.split("+")
            assert ocr_engine in ocr_engines
        else:
            ocr_engine = method
            method = None
    elif "gpt-4o-vision" in mode:
        ocr_engine, method, gpt_model = (
            None,
            mode.replace("gpt-4o-vision", "all_images"),
            "gpt-4o",
        )
    else:
        logger.warning(f"Skipped {mode}")
        continue
    mode_tuples[mode] = (ocr_engine, method, gpt_model)


def get_mode_str(ocr_engine, method, gpt_model):
    out = f"{ocr_engine}+{method}->{gpt_model}"
    out = out.replace("None+", "").replace("+None", "").replace("->None", "")
    return out


# %% [markdown]
# Reshaping df to generate figure

cols_to_keep = [
    col
    for col in ["writer_id", "id1", "id2", "id3", "id3_list", "gt", "num_pages"]
    if col in df.columns
]


dfs_to_combine = []
for mode, (ocr_engine, method, gpt_model) in mode_tuples.items():
    df_ = df.copy()
    cols = df_.columns
    df_ = df_[cols_to_keep + [mode, f"{mode}_cost"]]
    df_["ocr_engine"] = ocr_engine
    df_["method"] = method
    df_["gpt_model"] = gpt_model
    df_["mode_name"] = get_mode_str(ocr_engine, method, gpt_model)
    df_ = df_.rename(columns={f"{mode}_cost": "cost", mode: "pred"})
    dfs_to_combine.append(df_)
df = pd.concat(dfs_to_combine).reset_index(names="doc_id")

# Join pages, dropping the first page if necessary
page_join = lambda x: str_proc(
    newpage.join(x[drop_first_page:])
    if isinstance(x, list)
    else newpage.join(x.split(str_proc(newpage))[drop_first_page:])
)

# Convert lists of strings to strings
df["pred"] = df["pred"].apply(page_join)
df["gt"] = df["gt"].apply(page_join)


# %% [markdown]
# Generate figure

df = df.fillna("None")

df_agg_score = (
    df.groupby(["mode_name", "ocr_engine", "method", "gpt_model"])
    .apply(lambda x: score(gt=x["gt"].tolist(), pred=x["pred"].tolist()))
    .to_frame(name=metric)
)
df_agg_cost = df.groupby(["mode_name", "ocr_engine", "method", "gpt_model"]).agg(
    {"cost": "sum"}
)
df_agg = pd.concat([df_agg_score, df_agg_cost], axis=1).reset_index()

ocr_strs = {
    "azure": "Azure",
    "google_ocr": "Google",
    "textract": "Textract",
    "None": "All/None",
}

# Mappings for methods (colors)
num_blues = 7
blues = [get_cmap("Blues", num_blues)(i) for i in range(num_blues)][1:]
greys = ["#BFBFBF", "#4F4F4F"]
color_mapping = {
    # ocr -> gpt with increasing complexity
    "None": blues[0],
    "per_page": blues[1],
    "page": blues[2],
    "gpt-4o-mini-chosen_page": blues[3],
    "all_pages": blues[4],
    "page_pbp": blues[5],
    # image only with increasing complexity
    "all_images": greys[0],
    "all_images-per_page": greys[1],
    # other
    "google_ocr+azure+textract": "green",
}


# Define arbitrary styling function
def get_marker_style(gpt_model, method, ocr_engine):
    """
    Determines marker style, color, and size based on gpt_model, method, and ocr_engine.

    Parameters:
        gpt_model (str): GPT model used.
        method (str): Method used.
        ocr_engine (str): OCR engine used.

    Returns:
        dict: A dictionary containing 'color', 'marker', and 'size' for styling.
    """
    # Mappings for OCR engines (markers)
    marker_mapping = {
        "azure": "s",  # Square
        "google_ocr": "^",  # Triangle
        "textract": "X",  # Cross
        "None": "o",  # Circle
    }

    # Mappings for GPT models (sizes)
    size_mapping = {
        "None": 40,
        "gpt-4o-mini": 70,
        "gpt-4o": 100,
    }

    # Get styles
    color = color_mapping.get(method, "black")
    marker = marker_mapping.get(ocr_engine, "x")
    size = size_mapping.get(gpt_model, 40)

    return {"color": color, "marker": marker, "size": size}


# Step 1: Sort by CER and cost
df_agg = df_agg.sort_values(by=["CER", "cost"], ascending=[True, True])

# Step 2: Identify Pareto frontier points
pareto_points = []
current_min_cost = float("inf")

for _, row in df_agg.iterrows():
    if row["cost"] < current_min_cost:
        pareto_points.append(row)
        current_min_cost = row["cost"]

# Convert Pareto points to a DataFrame
pareto_df = pd.DataFrame(pareto_points)

# Step 3: Plot all points with styling based on gpt_model, method, and ocr_engine
plt.figure(figsize=(5, 5))
for _, row in df_agg.iterrows():
    style = get_marker_style(row["gpt_model"], row["method"], row["ocr_engine"])
    plt.scatter(
        row["CER"],
        row["cost"],
        color=style["color"],
        marker=style["marker"],
        s=style["size"],
        alpha=0.9,
    )

# Step 4: Plot the Pareto frontier
plt.plot(
    pareto_df["CER"],
    pareto_df["cost"],
    color="gray",
    linestyle="--",
    alpha=0.7,
)

# Step 5: Add text for Pareto frontier points
txt_box = []
for _, row in pareto_df.iterrows():
    ocr_engine, method, gpt_model = row["ocr_engine"], row["method"], row["gpt_model"]
    mode_name_text = (
        f"{ocr_strs[ocr_engine] if ocr_engine != 'None' else ''} {method_strs[method]}"
        + (f" $\\rightarrow$ {gpt_model}" if gpt_model != "None" else "")
    )
    txt_box.append(mode_name_text)

    # Add a multi-line text box
    text = "\\textbf{Pareto frontier methods} ($L \\rightarrow R$):\n" + "\n".join(
        txt_box
    )

    plt.text(
        (0.007 + 0.075) / 2 + 0.015,
        4.5,
        text,
        # fontsize=12,
        ha="right",  # Center the text horizontally
        va="bottom",  # Align the bottom of the text with the y-coordinate
        bbox=dict(facecolor="lightgrey", edgecolor="black", boxstyle="round,pad=0.5"),
    )

# Create legend handles for OCR engines (markers)
ocr_engine_legend = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=10,
        label=ocr_strs[engine],
    )
    for engine, marker in {
        "azure": "s",
        "google_ocr": "^",
        "textract": "X",
        "None": "o",
    }.items()
]

# Create legend handles for methods (colors)
method_legend = [
    Line2D(
        [0], [0], color=color, marker="o", linestyle="None", markersize=10, label=method
    )
    for method, color in {method_strs[k]: v for k, v in color_mapping.items()}.items()
]

# Create legend handles for GPT models (sizes)
gpt_model_legend = [
    Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        linestyle="None",
        markersize=size / 10,
        label=model,
    )
    for model, size in {
        "None": 30,
        "gpt-4o-mini": 70,
        "gpt-4o": 110,
    }.items()
]

# Add legends to the plot
plt.legend(
    handles=ocr_engine_legend,
    title="OCR Engine",
    loc="upper right",
    bbox_to_anchor=(1, 1),
)
plt.gca().add_artist(
    plt.legend(
        handles=ocr_engine_legend,
        title="OCR Engine",
        loc="upper center",
        bbox_to_anchor=(0.475, 1),
    )
)
plt.gca().add_artist(
    plt.legend(
        handles=method_legend, title="Method", loc="upper right", bbox_to_anchor=(1, 1)
    )
)
plt.gca().add_artist(
    plt.legend(
        handles=gpt_model_legend,
        title="GPT Model",
        loc="lower right",
        bbox_to_anchor=(1, 0),
    )
)

# Add labels and grid
plt.xlabel("CER")
plt.ylabel("Cost (\$)")
plt.grid(True)
# plt.ylim(0, 4.5)
plt.xlim(0.007, 0.075)
plt.tight_layout()

# Save to pdf, cropped
plt.savefig(plots / f"{results_path.stem}_pareto.pdf", bbox_inches="tight")
plt.show()

# %%
