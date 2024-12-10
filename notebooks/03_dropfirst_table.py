# %% [markdown]
# # Table showing performance for pages **after the first page only**; for looking at first_page performance.
# Generates Table 2 in the paper.

from prompts import newpage
from judge_htr import results, data
import pandas as pd
from evaluate import load
from loguru import logger
from tqdm import tqdm
from judge_htr.postprocessing import (
    gpt_output_postprocessing,
    get_mode_elements,
    get_mode_str,
    ocr_strs_short,
    generate_colored_latex_table,
)
import anls_star
from diskcache import Cache
import matplotlib.pyplot as plt
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
drop_first_page = True
remove_whitespace = False
seed = 0

# IAM (2 pages)
split = 0.5
results_file = f"iam_multipage_minpages=02_split={split:.02f}_seed={seed:02d}"
results_path = results / f"{results_file}_checked_strict.pkl"
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
# ### Generate LaTeX tables

ocr_engines = [
    "azure",
]

latex_tables = []

for engine in ocr_engines:
    d = df_agg.copy()
    d = d[(d["ocr_engine"] == engine) | (d["ocr_engine"] == "None")]

    # Filter only the methods we want in the table
    d = d[d["method"].apply(lambda s: s in ["None", "first_page"])]

    relative_increase = lambda baseline, new: (baseline - new) / baseline

    ocr_engine_score = d[d["mode_name"] == engine][metric].values[0]
    d["relative_improvement"] = (
        d[metric]
        .apply(lambda mode_score: relative_increase(ocr_engine_score, mode_score))
        .round(2)
    )
    ocr_baseline_cost = d[d["mode_name"] == engine]["cost"].values[0]

    label = f"{engine}_improvement_{results_path.stem}"
    caption = (
        "Relative performance of MLLMs and prompting strategies "
        f"compared to the baseline \\textbf{{{ocr_strs_short[engine]}}} "
        "OCR engine on the \\texttt{IAM} dataset. "
        "`pbp` denotes `page-by-page' and `*' denotes that no OCR engine was used. "
        "Rows are ordered by total cost of processing the dataset. "
        "Performance is given \\textit{only for pages after the first.}"
    )

    # Sort by cols: gpt_model, then method, then cost
    d = d.sort_values(by=["cost", "method"])

    logger.info(f"\n\n{engine}:\n{d}")

    latex_table = generate_colored_latex_table(
        df=d,
        metric_col=metric,
        improvement_cols={
            "relative_improvement": " \\makecell{Rel.\\\\Imp.}",
        },
        label=label,
        caption=caption,
    )
    latex_tables.append(latex_table)

pyperclip.copy("\n\n\n".join(latex_tables))
logger.info(f"\nCopied LaTeX tables to clipboard for {', '.join(ocr_engines)}")

# %% [markdown]
