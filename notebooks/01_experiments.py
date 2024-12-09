# %% [markdown]
# # Multi-page transcription experiments notebook

experiment_mode = "iam_multipage_minpages=02"

# # For 03_pagecount_plot.py
# experiment_mode = "iam_multipage_2-10_pages_10_docs"

# %% [markdown]
# ### Setup
from judge_htr import data, configs, results
from loguru import logger
from tqdm import tqdm
import pandas as pd
from judge_htr.gpt.caller import OpenAIGPT, api_call
from pathlib import Path
from dotenv import load_dotenv
from prompts import newpage, prompts, ocr_engine_replaceable_str
from judge_htr.postprocessing import (
    gpt_output_postprocessing as str_proc,
    ocr_strs,
)
from typing import Optional
from omegaconf import OmegaConf
import sys

load_dotenv()
tqdm.pandas()
args = {
    k: v for k, v in zip(sys.argv[1::2], sys.argv[2::2])
}  # command line args - override config/defaults

try:
    logger.level("COST", no=35, color="<yellow>")
except:
    logger.info("Logger already set up.")

# OCR costs $ / doc
ocr_costs = {
    "google_ocr": 1.5e-3,
    "azure": 1e-3,
    "textract": 1.5e-3,
}

experiment_mode = args.get("experiment_mode", experiment_mode)
cfg = dict(OmegaConf.load(configs / f"{experiment_mode}.yaml")) | args
logger.info("\n" + OmegaConf.to_yaml(cfg))


# Default OpenAI GPT model
model = cfg["gpt_model"]
# OCR engines to use
ocr_engines = cfg["ocr_engines"]

gpt = OpenAIGPT(
    model=model,
    api_call=api_call,
    verbose=False,
    cache_path=(data / cfg["gpt_cache_path"]),
    update_cache_every_n_calls=5,
)

api_call_kwargs = {
    "logprobs": True,
    "top_logprobs": 5,
    "seed": cfg["seed"],
}


# %% [markdown]
# ### Load dataset and set split

dataset_path = data / cfg["dataset_dir"] / f"{experiment_mode}.pkl"
split, seed = cfg["split"], cfg["seed"]
df = pd.read_pickle(data / dataset_path)
logger.info(f"\nLoaded {len(df)} docs from {dataset_path}")

assert df.index.is_unique, "Index not unique"

df = df.sample(frac=split, random_state=seed)
logger.info(f"\nUsing split of size {split} ({len(df)} docs) for experimentation")

page_counts = df["gt"].apply(len)
logger.info(f"\nDistribution of page counts:\n{page_counts.value_counts()}")

# Set up output files and batch API files if necessary
output_filename = f"{experiment_mode}_{model}_split={split:.2f}_seed={seed:02d}"
if cfg["batch_api"]:
    batch_api_dir = data / "batch_api"
    batch_api_dir.mkdir(exist_ok=True)
    batch_api_filepath = batch_api_dir / (output_filename + "_batch_api.jsonl")
    # Reset files
    with open(batch_api_filepath, "w") as f:
        f.write("")
    with open(batch_api_filepath.with_suffix(".hash"), "w") as f:
        f.write("")
    batch_api_custom_id = output_filename
    logger.info(
        f"\nSaving calls to batch API file with custom ID: {batch_api_custom_id}"
    )
else:
    batch_api_filepath, batch_api_custom_id = None, None
output_filename += ".pkl"

# %% [markdown]
# Now that we have a bunch of multipage docs,
# we can start to look at varying combinations of OCR text, text-only LLMs and vision LLMs
# for multi-page document transcription.

all_modes = []

# %% [markdown]
# ### GPT on raw OCR outputs only, whole document


def extract(
    ocr_text: list[str],
    prompt: str,
    img_paths: Optional[list[Path]] = None,
    batch_api_custom_id: Optional[str] = None,
) -> tuple[str, float]:
    """Returns both the output from a GPT call and its cost"""
    res = gpt.call(
        user_prompt=newpage.join(ocr_text),
        system_prompt=prompt,
        image_paths=img_paths,
        batch_api_filepath=batch_api_filepath,
        batch_api_custom_id=batch_api_custom_id,
        **api_call_kwargs,
    )
    cost = (
        gpt.cost_tracker.calculate_cost(
            input_tokens=res["usage"]["prompt_tokens"],
            output_tokens=res["usage"]["completion_tokens"],
        )
        if "usage" in res
        else 0.0
    )
    return str_proc(res["response"]), cost


for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}->{gpt.model}"
    prompt = prompts["ocr_only"].replace(
        ocr_engine_replaceable_str,
        f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
    )
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: extract(
            ocr_text=row[ocr_engine],
            prompt=prompt,
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)


# %% [markdown]
# ### GPT on raw OCR outputs only, one page at a time


def extract_pbp(
    ocr_text: list[str],
    prompt: str,
    img_paths: Optional[list[Path]] = None,
    batch_api_custom_id: Optional[str] = None,
) -> tuple[str, float]:
    """Returns both the output from a GPT call and its cost, processing PAGE-BY-PAGE."""
    output, cost = [], 0.0
    for i, page in enumerate(ocr_text):
        batch_api_custom_id = (
            (batch_api_custom_id + f"_page_{i:02d}") if batch_api_custom_id else None
        )
        res = gpt.call(
            user_prompt=page,
            system_prompt=prompt,
            image_paths=([img_paths[i]] if img_paths else None),
            batch_api_filepath=batch_api_filepath,
            batch_api_custom_id=batch_api_custom_id,
            **api_call_kwargs,
        )
        output.append(res["response"])
        cost += (
            gpt.cost_tracker.calculate_cost(
                input_tokens=res["usage"]["prompt_tokens"],
                output_tokens=res["usage"]["completion_tokens"],
            )
            if "usage" in res
            else 0.0
        )

    return str_proc(newpage.join(output)), cost


for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+pbp->{gpt.model}"
    prompt = prompts["ocr_only_pbp"].replace(
        ocr_engine_replaceable_str,
        f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
    )
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: extract_pbp(
            ocr_text=row[ocr_engine],
            prompt=prompt,
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)


# %% [markdown]
# ### OCR + page -> GPT, page-by-page

gpt.model = "gpt-4o"  # -mini inflates image tokens so can exceed limit

for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+all_pages_pbp->{gpt.model}"
    prompt = prompts["all_pages_pbp"].replace(
        ocr_engine_replaceable_str,
        f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
    )
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: extract_pbp(
            ocr_text=row[ocr_engine],
            prompt=prompt,
            img_paths=row["img_path"],
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)

gpt.model = model  # reset

# %% [markdown]
# ### Giving the **first page** as an image to GPT-4o and asking it to convert the rest from OCR text


for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+first_page->{gpt.model}"
    prompt = prompts["first_page"].replace(
        ocr_engine_replaceable_str,
        f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
    )
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: extract(
            ocr_text=row[ocr_engine],
            prompt=prompt,
            img_paths=row["img_path"][:1],
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)


# %% [markdown]
# ### Allow GPT to choose which page to look at based on the OCR output.

page_id_gpt_model = "gpt-4o-mini"
gpt.model = page_id_gpt_model

PAGE_IDS_FLAG = True

def choose_page_id(
    ocr_text: list[str],
    ocr_engine: str,
    batch_api_custom_id: Optional[str] = None,
) -> tuple[int, float]:
    """Returns a choice of page ID from a GPT call and the cost of the call."""
    prompt = prompts["choose_page_id"].replace(
        ocr_engine_replaceable_str,
        f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
    )
    res = gpt.call(
        user_prompt=newpage.join(ocr_text),
        system_prompt=prompt,
        batch_api_filepath=batch_api_filepath,
        batch_api_custom_id=batch_api_custom_id,
        **api_call_kwargs,
    )
    # Post-process and check page id str->int
    page_id = res["response"]
    if page_id.isdigit() and 0 < int(page_id) <= len(ocr_text):
        page_id = int(page_id)
    elif res.get("batch_call", False) == True:
        # Don't run chosen_page if we don't have page IDs for the prompts
        global PAGE_IDS_FLAG
        PAGE_IDS_FLAG = False 
        page_id = -1
    else:
        logger.warning(f"\nInvalid page ID:\n{page_id}")
        page_id = -1  # will ultimately default to 1, but flags a failed call

    cost = (
        gpt.cost_tracker.calculate_cost(
            input_tokens=res["usage"]["prompt_tokens"],
            output_tokens=res["usage"]["completion_tokens"],
        )
        if "usage" in res
        else 0.0
    )

    return page_id, cost


for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}_page_id"
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: choose_page_id(
            ocr_text=row[ocr_engine],
            ocr_engine=ocr_engine,
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    df[mode] = df[mode].astype(int)
gpt.model = model  # reset

# Give GPT OCR text and chosen page
if PAGE_IDS_FLAG == False:
    logger.warning(
        "\n\nSkipping chosen_page as no page IDs were found in the prompt. "
        "(Likely because calls were written to batch file.)"
    )
for ocr_engine in (ocr_engines if PAGE_IDS_FLAG else []):
    mode = f"{ocr_engine}+chosen_page->{gpt.model}"
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: extract(
            ocr_text=row[ocr_engine],
            prompt=prompts["chosen_page"](
                page_id := (
                    row[f"{ocr_engine}_page_id"]
                    if row[f"{ocr_engine}_page_id"] > 0
                    else 1
                )
            ).replace(
                ocr_engine_replaceable_str,
                f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
            ),
            img_paths=[row["img_path"][page_id - 1]],
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_page={page_id:02d}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    # Add page_id request cost to transcription cost
    df[mode + "_cost"] += df[f"{ocr_engine}_page_id_cost"]
    df.drop(columns=[f"{ocr_engine}_page_id_cost"], inplace=True)

    all_modes.append(mode)


# %% [markdown]
# ### Giving all the pages and the OCR text to GPT

gpt.model = "gpt-4o"  # -mini inflates image tokens so can exceed limit

for ocr_engine in ocr_engines:
    mode = f"{ocr_engine}+all_pages->gpt-4o"
    prompt = prompts["all_pages"].replace(
        ocr_engine_replaceable_str,
        f"{ocr_engine_replaceable_str} processed by {ocr_strs[ocr_engine]}",
    )
    logger.info(f"\n\nProcessing {mode}...")
    df[[mode, mode + "_cost"]] = df.progress_apply(
        lambda row: extract(
            ocr_text=row[ocr_engine],
            prompt=prompt,
            img_paths=row["img_path"],
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
                if batch_api_custom_id
                else None
            ),
        ),
        axis=1,
        result_type="expand",
    )
    all_modes.append(mode)

gpt.model = model  # reset


# %% [markdown]
# ### Full GPT-4o-vision on all pages without OCR text

gpt.model = "gpt-4o"  # -mini inflates image tokens so can exceed limit

mode = "vision*->gpt-4o"
logger.info(f"\n\nProcessing {mode}...")
df[[mode, mode + "_cost"]] = df.progress_apply(
    lambda row: extract(
        ocr_text="",
        prompt=prompts["vision*"],
        img_paths=row["img_path"],
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
    ),
    axis=1,
    result_type="expand",
)
all_modes.append(mode)

gpt.model = model  # reset

# %% [markdown]
# ### Full GPT-4o-vision on all pages INDIVIDUALLY without OCR text

gpt.model = "gpt-4o"  # -mini inflates image tokens so can exceed limit

mode = "vision*_pbp->gpt-4o"
logger.info(f"\n\nProcessing {mode}...")


def extract_pbp_vision(row: pd.Series) -> tuple[str, float]:
    """Extracts GPT Vision output PAGE-BY-PAGE and combines across pages."""
    cost, res = 0.0, []
    for i, img_path in enumerate(row["img_path"]):
        res_, cost_ = extract(
            ocr_text="",
            prompt=prompts["vision*_pbp"],
            img_paths=[img_path],
            batch_api_custom_id=(
                f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}_page_{i:02d}"
                if batch_api_custom_id
                else None
            ),
        )
        res.append(res_)
        cost += cost_
    return str_proc(newpage.join(res)), cost


df[[mode, mode + "_cost"]] = df.progress_apply(
    extract_pbp_vision,
    axis=1,
    result_type="expand",
)
all_modes.append(mode)

gpt.model = model  # reset


# %% [markdown]
# ### Combining OCR engines (one page at a time)


def extract_combined_ocrs(
    ocr_texts: dict[str, list[str]],
    prompt: str,
    img_paths: Optional[list[Path]] = None,
    batch_api_custom_id: Optional[str] = None,
) -> tuple[str, float]:
    """Returns both the output from a GPT call and its cost"""
    output, cost = [], 0.0

    for page_id in range(len(ocr_texts[ocr_engines[0]])):
        input = []
        for engine, ocr_text in ocr_texts.items():
            engine_marker = f"OCR_OUTPUT_{ocr_strs[engine].upper()}".replace(" ", "_")
            input.append(f"[{engine_marker}]\n{ocr_text[page_id]}\n[/{engine_marker}]")
        batch_api_custom_id = (
            (batch_api_custom_id + f"_page_{page_id:02d}")
            if batch_api_custom_id
            else None
        )
        res = gpt.call(
            user_prompt="\n\n".join(input),
            system_prompt=prompt,
            image_paths=img_paths,
            batch_api_filepath=batch_api_filepath,
            batch_api_custom_id=batch_api_custom_id,
            **api_call_kwargs,
        )
        output.append(res["response"])
        cost += (
            gpt.cost_tracker.calculate_cost(
                input_tokens=res["usage"]["prompt_tokens"],
                output_tokens=res["usage"]["completion_tokens"],
            )
            if "usage" in res
            else 0.0
        )

    return str_proc(newpage.join(output)), cost


combined_ocr_engines = list(ocr_costs.keys())

mode = f"{'_'.join(combined_ocr_engines)}_pbp->{gpt.model}"
prompt = prompts["all_ocr_pbp"]

logger.info(f"\n\nProcessing {mode}...")
df[[mode, mode + "_cost"]] = df.progress_apply(
    lambda row: extract_combined_ocrs(
        ocr_texts={engine: row[engine] for engine in combined_ocr_engines},
        prompt=prompt,
        batch_api_custom_id=(
            f"{batch_api_custom_id}_{mode}_id={int(row.name):04d}"
            if batch_api_custom_id
            else None
        ),
    ),
    axis=1,
    result_type="expand",
)
all_modes.append(mode)


# %% [markdown]
# ### Aggregating cost sources

for ocr_engine in combined_ocr_engines:
    df[ocr_engine + "_cost"] = df[ocr_engine].apply(
        lambda x: len(x) * ocr_costs[ocr_engine]
    )
    for mode in all_modes:  # GPT costs
        if ocr_engine in mode:
            df[mode + "_cost"] += df[ocr_engine + "_cost"]


# %% [markdown]
# ### Save outputs of experiments for eval in a separate notebook

df.to_pickle(results / output_filename)
logger.info(f"\nSaved outputs to {results / output_filename}")

gpt.update_cache()

# %% [markdown]
# gpt.backup_cache()
