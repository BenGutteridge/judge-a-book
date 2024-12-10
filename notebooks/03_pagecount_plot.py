# %% [markdown]
# # Generate plot of performance against page count. Corresponds to Figure 7 in the paper.

from prompts import newpage
from judge_htr import results, data, plots
import pandas as pd
from evaluate import load
from loguru import logger
from tqdm import tqdm
from judge_htr.postprocessing import (
    gpt_output_postprocessing,
    get_mode_elements,
    get_mode_str,
    ocr_strs_short,
)
import anls_star
from diskcache import Cache
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import os

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
split = 1.0
results_file = f"iam_multipage_2-10_pages_10_docs_split={split:.02f}_seed={seed:02d}"
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
df["num_pages"] = df["gt"].apply(len)
df["pred"] = df["pred"].apply(page_join)
df["gt"] = df["gt"].apply(page_join)

df = df.fillna("None")

# %% [markdown]
# ### Apply metric and aggregate scores and costs over modes and page counts
to_cat = []
for n in range(2, 11):
    df_agg_score = (
        df[df["num_pages"] == n]
        .groupby(["mode_name", "ocr_engine", "method", "gpt_model", "num_pages"])
        .apply(lambda x: score(gt=x["gt"].tolist(), pred=x["pred"].tolist()))
        .reset_index()  # Flatten the index
        .rename(columns={0: f"CER_{n}"})  # Rename the column to CER_n
    )
    ocr_engine_score = df_agg_score[df_agg_score["mode_name"] == "azure"][
        f"CER_{n}"
    ].values[0]
    df_agg_score[f"CER_improvement_{n}"] = df_agg_score[f"CER_{n}"].apply(
        lambda mode_score: (ocr_engine_score - mode_score) / ocr_engine_score
    )
    to_cat.append(df_agg_score[[f"CER_improvement_{n}"]])

# Concatenate all CER_n columns into a single DataFrame
df = pd.concat(to_cat, axis=1)

# Add back the identifying columns from the first iteration
identifying_columns = df_agg_score[
    ["mode_name", "ocr_engine", "method", "gpt_model", "num_pages"]
]
df = pd.concat([identifying_columns, df], axis=1)

# Final reshaping
cer_columns = [col for col in df.columns if col.startswith("CER_improvement_")]
df_long = df.melt(
    id_vars=["mode_name", "ocr_engine", "method", "gpt_model", "num_pages"],
    value_vars=cer_columns,
    var_name="page_count",
    value_name="CER_improvement",
)

# %% [markdown]
# Generate and save plot

plt.figure(figsize=(4, 3))
mode_names = df_long["mode_name"].unique()

# Custom styling
num_blues = 6
blues = [get_cmap("Blues", num_blues)(i) for i in range(num_blues)]
mode_names_style = {
    "azure->gpt-4o": {
        "name": r"\textsc{ocr only}",
        "linestyle": {"linestyle": "-", "color": blues[3], "marker": "s"},
        "order": 0,
    },
    "azure+pbp->gpt-4o": {
        "name": r"\textsc{ocr only pbp}",
        "linestyle": {"linestyle": "--", "color": blues[3], "marker": "s"},
        "order": 1,
    },
    "azure+all_pages->gpt-4o": {
        "name": r"\textsc{+all pages}",
        "linestyle": {"linestyle": "-", "color": blues[5], "marker": "D"},
        "order": 2,
    },
    f"azure+all_pages_pbp->gpt-4o": {
        "name": r"\textsc{{+all pages pbp}",
        "linestyle": {"linestyle": "--", "color": blues[5], "marker": "D"},
        "order": 3,
    },
    "vision*->gpt-4o": {
        "name": r"\textsc{vision*}",
        "linestyle": {"linestyle": "-", "color": "gray", "marker": "o"},
        "order": 4,
    },
    "vision*_pbp->gpt-4o": {
        "name": r"\textsc{vision* pbp}",
        "linestyle": {"linestyle": "--", "color": "gray", "marker": "o"},
        "order": 5,
    },
}

# Plot
for mode in sorted(mode_names_style.keys(), key=lambda s: mode_names_style[s]["order"]):
    subset = df_long[df_long["mode_name"] == mode]
    ocr_engine = subset["ocr_engine"].iloc[0]
    method = subset["method"].iloc[0]
    gpt_model = subset["gpt_model"].iloc[0]

    plt.plot(
        subset["page_count"].apply(lambda s: int(s.split("_")[-1])),
        subset["CER_improvement"],
        label=mode_names_style[mode]["name"],
        **mode_names_style[mode]["linestyle"],
    )

plt.xlabel("Page Count")
plt.ylabel("Relative CER improvement")
plt.legend(loc="lower left")
plt.xticks(range(2, 11))
plt.axhline(0, color="gray", alpha=0.5, linestyle=":")
plt.tight_layout()

plt.savefig(plots / f"{results_path.stem}_per_pagecount.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
