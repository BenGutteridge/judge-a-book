"""
Having created batch files, this script uploads them to OpenAI,
submits them for processing, retrieves the results, and saves them to the GPT cache.

Afterwards you should be able to re-run the experiment notebook without the batch args,
and the calls in that notebook should load from the cache.

We use the batch API because (i) it is cheaper (~half the cost)
to submit OpenAI API calls as a batch, and (ii) batch jobs often finish 
faster as they are processed in parallel.

https://platform.openai.com/docs/guides/batch

Make sure to run this notebook one cell at a time and follow the markdown instructions in each cell, 
as some cells require you to wait for the batch to complete.
"""

# %%
from judge_htr import data
import json
from loguru import logger
from tqdm import tqdm
from datetime import datetime
from judge_htr.gpt.caller import OpenAIGPT, api_call
from openai import OpenAI
from pathlib import Path

try:
    logger.level("COST", no=35, color="<yellow>")
except:
    logger.info("Logger already set up.")

# %%

client = OpenAI()

# %% [markdown]
# ## 0. Example of a batch file — a jsonl file of messages
#
# ```
# requests = [
#     {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}},
#     {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}},
# ]
# ```

# %% [markdown]
# ## 1. Get batch file paths

batch_dir = data / "batch_api"
batch_filepaths = [
    p
    for p in batch_dir.iterdir()
    if p.suffix == ".jsonl" and not p.stem.endswith("results")
]

# %%
# ### 1.1 Sanity check, is it the number of lines you expect?


def count_lines(filename):
    with open(filename, "r") as f:
        for i, line in enumerate(f, start=1):
            pass
    return i


[count_lines(str(p)) for p in batch_filepaths]

# %% [markdown]
# ### 1.2 If files are too big, split up


def split_jsonl_and_hash_by_size(jsonl_file: Path, max_size_mb: int = 150):
    """
    Splits a JSONL file and its accompanying .hash file into subfiles, ensuring each subfile
    is no larger than max_size_mb.

    Args:
        jsonl_file (Union[str, Path]): Path to the JSONL file.
        max_size_mb (int): Maximum size of each subfile in MB.
    """
    jsonl_file = Path(jsonl_file)
    hash_file = jsonl_file.with_suffix(".hash")

    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file {jsonl_file} does not exist.")
    if not hash_file.exists():
        raise FileNotFoundError(f"Hash file {hash_file} does not exist.")

    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    output_dir = jsonl_file.parent / "split"
    output_dir.mkdir(exist_ok=True)
    stem = jsonl_file.stem
    jsonl_suffix = jsonl_file.suffix
    hash_suffix = hash_file.suffix

    part_num = 1
    current_size = 0
    jsonl_buffer = []
    hash_buffer = []

    with open(jsonl_file, "r") as jf, open(hash_file, "r") as hf:
        for jsonl_line, hash_line in zip(jf, hf):
            line_size = len(jsonl_line.encode("utf-8"))
            # If adding this line exceeds the limit, write current buffer to subfiles
            if current_size + line_size > max_size_bytes:
                jsonl_subfile = output_dir / f"{stem}_part{part_num:03d}{jsonl_suffix}"
                hash_subfile = output_dir / f"{stem}_part{part_num:03d}{hash_suffix}"

                with open(jsonl_subfile, "w") as jsf, open(hash_subfile, "w") as hsf:
                    jsf.writelines(jsonl_buffer)
                    hsf.writelines(hash_buffer)

                print(
                    f"Written: {jsonl_subfile} and {hash_subfile} "
                    f"({current_size / (1024 * 1024):.2f} MB)"
                )

                part_num += 1
                jsonl_buffer = []
                hash_buffer = []
                current_size = 0

            # Add the lines to the buffers
            jsonl_buffer.append(jsonl_line)
            hash_buffer.append(hash_line)
            current_size += line_size

        # Write any remaining lines to the last subfiles
        if jsonl_buffer:
            jsonl_subfile = output_dir / f"{stem}_part{part_num:03d}{jsonl_suffix}"
            hash_subfile = output_dir / f"{stem}_part{part_num:03d}{hash_suffix}"

            with open(jsonl_subfile, "w") as jsf, open(hash_subfile, "w") as hsf:
                jsf.writelines(jsonl_buffer)
                hsf.writelines(hash_buffer)

            print(
                f"Written: {jsonl_subfile} and {hash_subfile} "
                f"({current_size / (1024 * 1024):.2f} MB)"
            )


def check_model_consistency_and_split_with_removal(
    jsonl_path: str,
    hash_path: str,
    output_inconsistent_jsonl: str,
    output_inconsistent_hash: str,
):
    """
    Checks if the "body"["model"] field is consistent across all lines in a JSONL file.
    If inconsistent lines are found, they and their corresponding hash lines are written to separate files,
    and they are removed from the original files.

    Parameters:
        jsonl_path (str): Path to the JSONL file.
        hash_path (str): Path to the corresponding .hash file.
        output_inconsistent_jsonl (str): Path to save the inconsistent JSONL lines.
        output_inconsistent_hash (str): Path to save the corresponding inconsistent .hash lines.

    Returns:
        bool: True if the "body"["model"] is consistent across all lines, False otherwise.
    """
    model_set = set()
    inconsistent_jsonl_lines = []
    inconsistent_hash_lines = []
    consistent_jsonl_lines = []
    consistent_hash_lines = []

    with open(jsonl_path, "r") as jsonl_file, open(hash_path, "r") as hash_file:
        for jsonl_line, hash_line in zip(jsonl_file, hash_file):
            try:
                data = json.loads(jsonl_line.strip())
                model = data.get("body", {}).get("model")
                if model:
                    model_set.add(model)
                    # Classify lines based on consistency
                    if len(model_set) > 1:
                        inconsistent_jsonl_lines.append(jsonl_line.strip())
                        inconsistent_hash_lines.append(hash_line.strip())
                    else:
                        consistent_jsonl_lines.append(jsonl_line.strip())
                        consistent_hash_lines.append(hash_line.strip())
                else:
                    print(f"Warning: Missing 'model' in line: {jsonl_line.strip()}")
                    inconsistent_jsonl_lines.append(jsonl_line.strip())
                    inconsistent_hash_lines.append(hash_line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {jsonl_line.strip()} - {e}")
                inconsistent_jsonl_lines.append(jsonl_line.strip())
                inconsistent_hash_lines.append(hash_line.strip())

    # Write inconsistent lines to separate files
    if inconsistent_jsonl_lines:
        print(f"Inconsistent models found: {model_set}")

        with open(output_inconsistent_jsonl, "w") as out_jsonl_file:
            for line in inconsistent_jsonl_lines:
                out_jsonl_file.write(line + "\n")

        with open(output_inconsistent_hash, "w") as out_hash_file:
            for line in inconsistent_hash_lines:
                out_hash_file.write(line + "\n")

        # Write back consistent lines to the original files
        with open(jsonl_path, "w") as jsonl_file:
            for line in consistent_jsonl_lines:
                jsonl_file.write(line + "\n")

        with open(hash_path, "w") as hash_file:
            for line in consistent_hash_lines:
                hash_file.write(line + "\n")

        print(
            f"Inconsistent JSONL lines have been written to {output_inconsistent_jsonl}"
        )
        print(
            f"Inconsistent HASH lines have been written to {output_inconsistent_hash}"
        )
        return False

    # All lines are consistent
    if model_set:
        print(f"All lines have the same model: {model_set.pop()}")
    else:
        print("No model found in the file.")
    return True


# Adds to subdir /split
for p in batch_filepaths:
    split_jsonl_and_hash_by_size(p, 150)

# %% [markdown]
# ### 1.3 Redo get batch_filepaths and sanity check with splits

sort_key = lambda p: ("mini" in p.name.lower(), p.name.lower())

batch_dir = batch_dir / "split"
batch_filepaths = sorted(
    [
        p
        for p in batch_dir.iterdir()
        if p.suffix == ".jsonl" and not p.stem.endswith("results")
    ],
    key=sort_key,
)

for p in batch_filepaths:
    check_model_consistency_and_split_with_removal(
        jsonl_path=p,
        hash_path=Path(str(p).replace(".jsonl", ".hash")),
        output_inconsistent_jsonl=Path(str(p).replace(".jsonl", "_inconsistent.jsonl")),
        output_inconsistent_hash=Path(str(p).replace(".jsonl", "_inconsistent.hash")),
    )

# Sanity check
[count_lines(str(p)) for p in batch_filepaths]

# %% [markdown]
# ## 2. Upload batch files and submit to run

for batch_filepath in tqdm(batch_filepaths):

    try:

        # Uploading batch input file
        logger.info(f"\nUploading {batch_filepath}...")
        batch_input_file = client.files.create(
            file=open(batch_filepath, "rb"),
            purpose="batch",
        )

        # Submitting the batch job
        logger.info(f"\nSubmitting batch for {batch_filepath}...")
        client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": batch_filepath.stem,
            },
        )

    except:
        logger.exception(f"Error with {batch_filepath}")
        continue


# %% [markdown]
# ## 3. Checking status
# Do not run cells below this one until the batches are complete

# Will print out the in_progress batches
logger.info("\nIn progress:")
count = 0
for x in client.batches.list(limit=100).data:
    desc = x.metadata["description"]
    s = f"\n{desc} {x.status} {datetime.fromtimestamp(x.created_at)}"
    if "in_progress" in s:
        logger.info(s)
        count += 1

# See the status of all batches
logger.info("\nAll batches:")
for x in client.batches.list(limit=100).data:
    desc = x.metadata["description"]
    logger.info(f"\n{desc} {x.status} {datetime.fromtimestamp(x.created_at)}")

logger.info(f"\n{count} in progress")

# %% [markdown]
# ## 4. Retrieving results and writing to a file
# Do not run this cell or the ones below it until the batches are complete
completed = set()
for x in tqdm(
    client.batches.list(limit=100).data
):  # use arg `after={final batch id}` to get >100
    dt = datetime.fromtimestamp(x.created_at)
    if (
        str(x.status)
        != "completed"
        #  or dt < datetime(2024, 12, 4): # fill as required
    ):
        continue
    res = client.files.content(x.output_file_id)
    desc = x.metadata["description"]
    results_path = batch_dir / f"{desc}_results.jsonl"
    with open(results_path, "w") as f:
        f.write(res.text)
    logger.info(f"\nWritten to {results_path}")
    completed.add(results_path.name.replace("_results.jsonl", ""))


# %% [markdown]
# ### 4.1 Check if all files are completed

logger.info("\nNot completed:")
not_completed = set(b.stem for b in batch_filepaths) - completed
logger.info(f"\n{not_completed}")


# %% [markdown]
# ## 5. Load results from batch calls and save them to the GPT cache

gpt = OpenAIGPT(
    model="gpt-4o-mini",  # arbitrary, model is not needed when saving to cache
    api_call=api_call,
)  # only instantiating in order to use gpt class methods

for batch_filepath in tqdm(batch_filepaths):

    if batch_filepath.stem in not_completed:
        logger.info(f"\nSkipping {batch_filepath}")
        continue

    res_filepath = batch_filepath.with_stem(batch_filepath.stem + "_results")
    hash_filepath = batch_filepath.with_suffix(".hash")

    # jsonl file to a list of dicts
    with open(res_filepath, "r") as f:
        results = [json.loads(line) for line in f]

    with open(hash_filepath, "r") as f:
        hashes = [json.loads(line) for line in f]

    count = len(hashes)
    hashes = {id: h for line in hashes for id, h in line.items()}
    assert len(hashes) == count, "Duplicate hashes"

    for r in tqdm(results):
        assert r["error"] is None, f"Error: {r['error']}"
        res_hash = hashes[r["custom_id"]]
        response = r["response"]["body"]

        if "iam" in r["custom_id"]:
            cache_path = data / "gpt_cache_iam.db"
        else:
            raise ValueError("Unknown dataset")

        if gpt.cache_path != cache_path:
            gpt = OpenAIGPT(
                model="gpt-4o-mini",
                api_call=api_call,
                cache_path=cache_path,
            )
            logger.info(f"\nSwitched cache to {cache_path}.")

        res = gpt.response_to_dict(response)

        gpt.add_to_cache(input_hash=res_hash, res=res)
        gpt.update_cache()

# %% [markdown]
# Now you can re-run 01_experiments.py without the batch flag and the results should already be cached.
