# %% [markdown]
# # Error catching
# Notebook that takes the output from the 01_experiments.py notebook and
# catches errors known to be drastic outliers.
# Removes:
# - Pages where the CER between the OCR and the post-processed GPT is high (>=.5)
# - Pages where the number of newpage markers is incorrect
# - Pages where the CER between the OCR and the post-processed GPT is high for
# any of the pages split by newpage markers
# - Pages where a newpage marker has been added right at the start of the output
#
# This should catch egregious errors, such as:
# - GPT not outputting anything ("I'm sorry, but..."")
# - GPT repeating blocks of text
# - GPT failing to interpret newpage markers (often a sign of other failures)
# - Misc GPT error
#
# Note that 'checked' results are used in all subsequent plot-/figure-generating notebooks,
# except for 03_dropfirst_table, since it uses only results for pages >= 2, so the page break checks must
# be reliable.

from prompts import newpage
from judge_htr import results
import pandas as pd
from judge_htr.postprocessing import (
    gpt_output_postprocessing,
    drop_duplicate_columns,
)
from loguru import logger
import numpy as np
from evaluate import load
import sys

args = {
    k: v for k, v in zip(sys.argv[1::2], sys.argv[2::2])
}  # command line args - override config/defaults
mode = args.get("mode", None)


# %% [markdown]
# ### Load results

if mode == "pagecount":
    gpt_model = "gpt-4o"
    exp_file = f"iam_multipage_2-10_pages_10_docs_{gpt_model}_split=1.00_seed=00.pkl"
    df = pd.read_pickle(results / exp_file)
else:
    for i, gpt_model in enumerate(["gpt-4o", "gpt-4o-mini"]):
        exp_file = f"iam_multipage_minpages=02_{gpt_model}_split=0.50_seed=00.pkl"
        if i == 0:
            df = pd.read_pickle(results / exp_file)
        else:
            df = pd.concat([df, pd.read_pickle(results / exp_file)], axis=1)

df = drop_duplicate_columns(df)
assert df.isna().sum().sum() == 0  # check correct concat

# %% [markdown]
# Save combined but unchecked df

# Save combined but unchecked df
output_file = (results / exp_file).with_name(
    (results / exp_file).stem.replace(f"{gpt_model}_", "") + ".pkl"
)
df.to_pickle(output_file)

# %% [markdown]
# ### Error catching setup

ocr_engines = [
    "azure",
    "google_ocr",
    "textract",
]

newpage = gpt_output_postprocessing(newpage)

cer = load("cer")


def cer_score(pred: str, gt: str) -> float:
    return cer.compute(predictions=[pred], references=[gt])


all_modes = [col.replace("_cost", "") for col in df.columns if col.endswith("_cost")]

mode_engine_pairs_to_check = {}  # only LLM modes which use OCR as input

for mode in all_modes:
    if mode in ocr_engines:
        continue
    for engine in ocr_engines:
        if engine in mode:
            mode_engine_pairs_to_check[mode] = engine

# Optional: add modes which use no OCR input at all (GPT-4o-vision). Off by default.
if default_engine := "":  # change to ""/None to turn off
    for mode in all_modes:
        if not any(engine in mode for engine in ocr_engines):
            mode_engine_pairs_to_check[mode] = default_engine

removed = {k: [] for k in mode_engine_pairs_to_check.keys()}

# %% [markdown]
# ### Newpage marker at start
logger.info("\n\nNewpage marker at start:")
for mode, engine in mode_engine_pairs_to_check.items():
    newpage_at_start = df[mode].apply(lambda s: newpage in s[:25])
    if newpage_at_start.sum() > 0:
        logger.info(f"\n{mode}: {newpage_at_start.sum()}")
        df[mode] = np.where(
            newpage_at_start, df[engine].apply(lambda s: newpage.join(s)), df[mode]
        )
        removed[mode] += df[mode][newpage_at_start].index.tolist()


# %% [markdown]
# ### Large CER gap with OCR engine
logger.info("\n\nLarge CER gap with OCR engine")
for mode, engine in mode_engine_pairs_to_check.items():
    cer_gap = df.apply(
        lambda row: cer_score(pred=row[mode], gt=newpage.join(row[engine])) > 0.4,
        axis=1,
    )
    if cer_gap.sum() > 0:
        logger.info(f"\n{mode}: {cer_gap.sum()}")
        df[mode] = np.where(
            cer_gap, df[engine].apply(lambda s: newpage.join(s)), df[mode]
        )
        removed[mode] += df[mode][cer_gap].index.tolist()


# %% [markdown]
# ### Save with basic, non-page-level error catching
for mode in mode_engine_pairs_to_check.keys():
    if len(removed[mode]) > 0:
        logger.info(f"\n{mode}: {len(removed[mode])}/{len(df)} removed")

df.to_pickle(output_file.with_stem(f"{output_file.stem}_checked"))


# %% [markdown]
# ### Incorrect page count
logger.info("\n\nIncorrect page counts:")
for mode, engine in mode_engine_pairs_to_check.items():
    engine_page_count = df[engine].apply(len)
    mode_page_count = df[mode].apply(lambda s: len(s.split(newpage)))
    incorrect_page_count = mode_page_count != engine_page_count
    if incorrect_page_count.sum() > 0:
        logger.info(f"\n{mode}: {incorrect_page_count.sum()}")
        df[mode] = np.where(
            incorrect_page_count,
            df[engine].apply(lambda s: newpage.join(s)),
            df[mode],
        )
        removed[mode] += df[mode][incorrect_page_count].index.tolist()


# %% [markdown]
# ### Large CER gap **within newpages**
logger.info("\n\nLarge CER gap within newpages")
for mode, engine in mode_engine_pairs_to_check.items():
    per_page_cer_gap = df.apply(
        lambda row: any(
            cer_score(
                pred=row[mode].split(newpage)[i],
                gt=row["gt"][i],
            )
            > 0.5
            for i in range(len(row["gt"]))
        ),
        axis=1,
    )
    if per_page_cer_gap.sum() > 0:
        logger.info(f"\n{mode}: {per_page_cer_gap.sum()}")
        # where x use y else z
        df[mode] = np.where(
            per_page_cer_gap, df[engine].apply(lambda s: newpage.join(s)), df[mode]
        )
        removed[mode] += df[mode][per_page_cer_gap].index.tolist()


# %% [markdown]
# ### Save with additional page-level error catching
for mode in mode_engine_pairs_to_check.keys():
    if len(removed[mode]) > 0:
        logger.info(f"\n{mode}: {len(removed[mode])}/{len(df)} removed")

df.to_pickle(output_file.with_stem(f"{output_file.stem}_checked_strict"))

# %% [markdown]
