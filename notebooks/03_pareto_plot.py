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
    get_mode_elements,
    get_mode_str,
    ocr_strs_short,
    gpt_model_strs,
)
import anls_star
from diskcache import Cache
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import get_cmap
import os
import pyperclip

os.environ["PATH"] += ":/Library/TeX/texbin"  # for latex in figures
from matplotlib import pyplot as plt


tqdm.pandas()
cache = Cache(data)
cer = load("cer")
plt.rcParams["text.usetex"] = True

tqdm.pandas()

# %% [markdown]
# ### Parameters and setup
drop_first_page = False
remove_whitespace = False
seed = 0

# IAM (2 pages)
split = 0.5
results_file = f"iam_multipage_minpages=02_split={split:.02f}_seed={seed:02d}"
results_path = results / f"{results_file}_checked.pkl"
df = pd.read_pickle(results_path)

# Metric
metric = "CER"  # Character Error Rate
# metric = "ANLS"  # ANLS*

score = {
    "CER": lambda gt, pred: cer.compute(predictions=pred, references=gt),
    "ANLS": lambda gt, pred: anls_star.anls_score(gt=gt, pred=pred),
}[metric]

# Param-dependent string manipulation utils
if remove_whitespace:
    str_proc = lambda s: gpt_output_postprocessing(s).replace(" ", "")
else:
    str_proc = gpt_output_postprocessing

# Concatenates list of page texts into single string with page breaks if a list, and removes first page if needed
page_join = lambda x: str_proc(
    newpage.join(x[drop_first_page:])
    if isinstance(x, list)
    else newpage.join(x.split(str_proc(newpage))[drop_first_page:])
)

# %% [markdown]
# ### Split up mode into OCR engine, method, and GPT model

all_modes = [col.replace("_cost", "") for col in df.columns if col.endswith("_cost")]
mode_tuples = {}
for mode in all_modes:
    ocr_engine, method, gpt_model = get_mode_elements(mode)
    mode_tuples[mode] = (ocr_engine, method, gpt_model)


# %% [markdown]
# ### Converting columns for each mode into rows

dfs_to_combine = []
for mode, (ocr_engine, method, gpt_model) in mode_tuples.items():
    df_ = df.copy()
    cols = df_.columns
    df_ = df_[["writer_id", "gt", mode, f"{mode}_cost"]]
    df_["ocr_engine"] = ocr_engine
    df_["method"] = method
    df_["gpt_model"] = gpt_model
    df_["mode_name"] = get_mode_str(ocr_engine, method, gpt_model)
    df_ = df_.rename(columns={f"{mode}_cost": "cost", mode: "pred"})
    dfs_to_combine.append(df_)
df = pd.concat(dfs_to_combine).reset_index(names="doc_id")

# Convert lists of strings to strings
df["pred"] = df["pred"].apply(page_join)
df["gt"] = df["gt"].apply(page_join)

df = df.fillna("None")

# %% [markdown]
# ### Apply metric and aggregate scores and costs over modes

df_agg_score = (
    df.groupby(["mode_name", "ocr_engine", "method", "gpt_model"])
    .apply(lambda x: score(gt=x["gt"].tolist(), pred=x["pred"].tolist()))
    .to_frame(name=metric)
)
df_agg_cost = df.groupby(["mode_name", "ocr_engine", "method", "gpt_model"]).agg(
    {"cost": "sum"}
)
df_agg = pd.concat([df_agg_score, df_agg_cost], axis=1).reset_index()

# Sort by CER and cost
df_agg = df_agg.sort_values(by=["CER", "cost"], ascending=[True, True])


# %% [markdown]
# ## Generate scatter plot figure with Pareto frontier

# %% [markdown]
# ### Define scatter plot styling function

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

# Mappings for methods (colors)
num_blues = 7
blues = [get_cmap("Blues", num_blues)(i) for i in range(num_blues)][1:]
greys = ["#BFBFBF", "#4F4F4F"]
color_mapping = {
    # ocr -> gpt with increasing complexity
    "None": blues[0],
    "pbp": blues[1],
    "first_page": blues[2],
    "chosen_page": blues[3],
    "all_pages": blues[4],
    "all_pages_pbp": blues[5],
    # image only with increasing complexity
    "vision*": greys[0],
    "vision*_pbp": greys[1],
    # other
    "google_ocr_azure_textract_pbp": "green",
}


def get_marker_style(gpt_model, method, ocr_engine):
    """Return dict of scatter marker elements based on method stages."""
    color = color_mapping[method]
    marker = marker_mapping[ocr_engine]
    size = size_mapping[gpt_model]
    return {"color": color, "marker": marker, "size": size}


# %% [markdown]
# ### Get subset of points that sit on the Pareto frontier

pareto_points = []
current_min_cost = float("inf")

for _, row in df_agg.iterrows():
    if row["cost"] < current_min_cost:
        pareto_points.append(row)
        current_min_cost = row["cost"]

pareto_df = pd.DataFrame(pareto_points)


# %% [markdown]
# ### Plot scatter plot of methods with Pareto frontier
plt.figure(figsize=(4.5, 3.5))

# Plot all points with styling based on gpt_model, method, and ocr_engine
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

# Frontier line
plt.plot(
    pareto_df["CER"],
    pareto_df["cost"],
    color="gray",
    linestyle="--",
    alpha=0.7,
)

# Get string for Pareto frontier methods
txt_box = "\\textbf{Pareto frontier methods} ($L \\rightarrow R$):"
for _, row in pareto_df.iterrows():
    ocr_engine = (
        ocr_strs_short[row["ocr_engine"]] if row["ocr_engine"] != "None" else ""
    )
    method = method_strs[row["method"]]
    gpt_model = (
        (" $\\rightarrow$ " + gpt_model_strs(row["gpt_model"]))
        if row["gpt_model"] != "None"
        else ""
    )
    txt_box += f"\n{ocr_engine} {method} {gpt_model}"

logger.info(
    f"\n\The following text listing Pareto frontier methods has been copied to clipboard:\n{txt_box}"
)
pyperclip.copy(txt_box)

# Create legend handles for OCR engines (markers)
ocr_engine_legend = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=10,
        label=ocr_strs_short[engine],
    )
    for engine, marker in marker_mapping.items()
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
        "None": 40,
        gpt_model_strs("gpt-4o-mini").replace("gpt-", ""): 70,
        gpt_model_strs("gpt-4o").replace("gpt-", ""): 110,
    }.items()
]

# Add legends to the plot
ocr_engine_legend = plt.legend(
    handles=ocr_engine_legend,
    title="OCR Engine",
    loc="upper right",
    bbox_to_anchor=(1, 1),
)

method_legend = plt.legend(
    handles=method_legend,
    title="Method",
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=3,
)

gpt_model_legend = plt.legend(
    handles=gpt_model_legend,
    title="GPT Model",
    loc="upper center",
    bbox_to_anchor=(0.5, 1),
)

plt.gca().add_artist(ocr_engine_legend)
plt.gca().add_artist(method_legend)
plt.gca().add_artist(gpt_model_legend)

# Add labels and grid
plt.xlabel("CER")
plt.ylabel("Cost (\$)")
plt.grid(True)
# plt.ylim(0, 4.5)
plt.xlim(0.007, 0.075)
plt.tight_layout()

# Save to pdf, cropped
plt.savefig(
    plots / f"{results_path.stem}_pareto.pdf",
    bbox_extra_artists=(ocr_engine_legend, method_legend, gpt_model_legend),
    bbox_inches="tight",
)
plt.tight_layout()
plt.show()

# %%
