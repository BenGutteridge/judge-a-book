# %% [markdown]
# # Main performance tables showing improvement over OCR engines wrt cost for various methods and MLLM prompting strategies

from prompts import newpage
from judge_htr import results, data
import pandas as pd
from judge_htr.preprocessing import gpt_output_postprocessing, ocr_strs
from evaluate import load
from loguru import logger
from tqdm import tqdm
import pyperclip
from judge_htr.postprocessing import (
    generate_colored_latex_table,
    ocr_strs,
)
import anls_star
from diskcache import Cache


tqdm.pandas()
cache = Cache(data)
cer = load("cer")

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

mode_tuples


def get_mode_str(ocr_engine, method, gpt_model):
    out = f"{ocr_engine}+{method}->{gpt_model}"
    out = out.replace("None+", "").replace("+None", "").replace("->None", "")
    return out


# %% [markdown]
# Building improved table

cols_to_keep = [
    col
    for col in ["writer_id", "id1", "id2", "id3", "id3_list", "gt"]
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
# Rearrange df

# Get scores and summed costs

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

df_agg = df_agg.sort_values(
    by=["ocr_engine", "gpt_model", "cost"],
    key=lambda col: col.map(
        lambda x: ocr_engines.index(x) if x in ocr_engines else len(ocr_engines)
    ),
).round({"cost": 2, metric: 3})

df_agg.drop(columns=["mode_name"])

# %% [markdown]
# Generate LaTeX tables

latex_tables = []

for engine in ocr_engines:
    d = df_agg.copy()
    d = d[(d["ocr_engine"] == engine) | (d["ocr_engine"] == "None")]

    relative_increase = lambda baseline, new: (baseline - new) / baseline

    ocr_engine_score = d[d["mode_name"] == engine][metric].values[0]
    d["relative_improvement"] = (
        d[metric]
        .apply(lambda mode_score: relative_increase(ocr_engine_score, mode_score))
        .round(2)
    )
    ocr_baseline_cost = d[d["mode_name"] == engine]["cost"].values[0]

    d["relative_improvement_over_cost_multiplier"] = (
        d["relative_improvement"] / (d["cost"] / ocr_baseline_cost)
    ).round(2)

    label = f"{engine}_improvement_{results_path.stem}"
    caption = (
        "Relative performance of MLLMs and prompting strategies "
        f"compared to the baseline \\textbf{{{ocr_strs[engine]}}} "
        f"OCR engine on the \\textbf{{{run_name}}} dataset. "
        "`pbp` denotes `page-by-page' and `*' denotes that no OCR engine was used. "
        "Rows are ordered by total cost of processing the dataset."
    )

    # Sort by cols: gpt_model, then method, then cost
    d = d.sort_values(by=["cost", "method"])

    logger.info(f"\n\n{engine}:\n{d}")

    latex_table = generate_colored_latex_table(
        df=d,
        metric_col=metric,
        improvement_cols={
            "relative_improvement": " \\makecell{Rel.\\\\Imp.}",
            "relative_improvement_over_cost_multiplier": "$\\dfrac{\\text{R.I.}}{c_\\text{cost}}$",
        },
        label=label,
        caption=caption,
    )
    latex_tables.append(latex_table)

pyperclip.copy("\n\n\n".join(latex_tables))

# %% [markdown]
