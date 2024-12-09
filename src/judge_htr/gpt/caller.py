"""
Generic GPT API caller class for aggregation experiments/notebooks.
Supports text and vision GPT OpenAI API calls.
Includes response processing, caching, and token limit handling.
"""

from judge_htr import data
from loguru import logger
from typing import Optional, NamedTuple
import openai
from openai import OpenAI
from pathlib import Path
from typing import Callable, Union
import json
import tiktoken
import base64
from PIL import Image
import shutil
import numpy as np
from judge_htr.gpt.cost_estimator import count_image_tokens, CostTracker
import sqlite3
from collections import defaultdict


MAX_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-3.5-turbo": 16_385,
}

default_cache_path = data / "gpt_cache.db"


def api_call(client, model, messages, **kwargs):
    """Wrapper for api call."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        **kwargs,
    )
    return response


class OpenAIGPT:
    def __init__(
        self,
        model: str,
        api_call: Callable,
        verbose: bool = False,
        cache_path: Path = default_cache_path,
        update_cache_every_n_calls: int = 10,
        backup_cache_every_n_calls: int = 50,
    ):
        self.cost_tracker = CostTracker(model=model)
        self.model = model
        self.api_call = self.cost_tracker(api_call)
        self.cache_path = cache_path
        self.cache_count = 0
        self.verbose = verbose
        self.update_cache_every_n_calls = update_cache_every_n_calls
        self.backup_cache_every_n_calls = backup_cache_every_n_calls

        # Initialize database for caching
        self.conn = sqlite3.connect(self.cache_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()

        self.client = OpenAI()

        self.default_system_prompt = (
            "You are ChatGPT, a large language model trained by OpenAI. "
            "Follow the user's instructions carefully. Respond using markdown."
        )  # default system prompt for chat interface

    def _initialize_database(self):
        """Create cache table if it doesn't exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gpt_cache (
                input_hash TEXT PRIMARY KEY,
                response_data TEXT
            )
        """
        )
        self.conn.commit()

    def call(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        image_paths: Optional[list[Path]] = None,
        load_from_cache: bool = True,
        batch_api_filepath: Optional[Path] = None,
        batch_api_custom_id: Optional[str] = None,
        auto_truncate: bool = False,
        **api_call_kwargs,
    ) -> Optional[str]:
        """Generic GPT call wrapper function.
        Includes the option to include images if image_paths is not None
        Includes the option to write API calls to a batch file instead of making the call directly
        (if batch_api_filepath and batch_api_custom_id are not None).
        NOTE that this only writes to the file, it does not submit the job.

        Args:
            user_prompt (str): The user prompt for the GPT call.
            system_prompt (str, optional): The system prompt for the GPT call. None defaults to the standard prompt.
            image_paths (list[Path], optional): List of image paths to include in the GPT call, if using GPT-vision.
            load_from_cache (bool, optional): Whether to load the GPT output from cache if available. Defaults to True.
            batch_api_filepath (Path, optional): Filepath to the batch API call file. Defaults to None.
            batch_api_custom_id (str, optional): Custom ID for the batch API call. Defaults to None.
            auto_truncate (bool, optional): Whether to automatically truncate the user prompt if it exceeds the token limit. Defaults to False.
            **api_call_kwargs: Additional keyword arguments for the API call.

        Returns:
            Optional[str]: The GPT response as a string, or None if writing calls to a batch file.
        """

        system_prompt = (
            system_prompt if system_prompt is not None else self.default_system_prompt
        )
        # Option to include images
        if image_paths:
            user_prompt = self.build_image_messages(image_paths) + (
                [user_prompt] if user_prompt else []
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        messages = self.check_if_exceeds_token_limit(
            messages=messages,
            image_paths=image_paths,
            auto_truncate=auto_truncate,
        )

        # Load from cache if available
        input_hash: str = self.api_input_to_hashable(messages, **api_call_kwargs)

        if load_from_cache and (res := self.load_from_cache(input_hash)) is not None:
            if res["probs"] is not None:
                res["probs"] = [TokProbs(tokens=t[0], probs=t[1]) for t in res["probs"]]
            if self.verbose:
                logger.info("Loaded GPT output from cache.")

        # Write to batch API call file if specified
        elif batch_api_filepath and batch_api_custom_id:
            logger.info(
                f"Appending call to batch API call file at {batch_api_filepath}"
            )
            self.save_to_batch_file(
                messages=messages,
                input_hash=input_hash,
                custom_id=batch_api_custom_id,
                batch_filepath=batch_api_filepath,
                **api_call_kwargs,
            )
            return defaultdict(
                lambda: None,  # will return None for all undefined keys
                {
                    "response": "Placeholder response; this call was written to a batch file.",
                    "cost": 0.0,
                    "batch_call": True,
                },
            )  # return a placeholder response that won't be cached or crash scripts

        # Otherwise, make API call
        else:
            try:
                response = self.api_call(
                    client=self.client,
                    model=self.model,
                    messages=messages,
                    **api_call_kwargs,
                )
                # Extract and restructure useful information from ChatCompletion object
                res = self.response_to_dict(response)
                self.add_to_cache(input_hash, res)

            except:
                logger.error("API call failed.")
                res = defaultdict(
                    lambda: None,
                    {
                        "response": "Placeholder response; API call failed.",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                        "cost": 0.0,
                    },
                )

        # Option to log prompt and response
        if self.verbose:
            prompt_to_log = (
                user_prompt
                if len(user_prompt) < 1000
                else user_prompt[:1000] + "\n...[truncated]"
            )
            response_to_log = (
                res["response"]
                if len(res["response"]) < 1000
                else res["response"][:1000] + "\n...[truncated]"
            )
            logger.info(prompt_to_log)
            logger.info(response_to_log)

        # Log cost, update cache if necessary, backup cache if necessary
        if (self.cache_count + 1) % self.update_cache_every_n_calls == 0:
            self.update_cache()

        return res

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        """If model is changed, update the cost tracker model."""
        self._model = value
        # Fine-tuned models are formatted as ft:<model>:<organization>:<ft-job-info>
        # We remove the org/job info and just keep ft:<model>
        value = ":".join(value.split(":")[:2])
        self.cost_tracker.model = value

    @property
    def tokenizer(self):
        try:
            self._tokenizer = tiktoken.encoding_for_model(
                self.cost_tracker.model.replace("ft:", "")
            )
        except:
            raise ValueError(f"Model {self.model} not supported?")
        return self._tokenizer

    @property
    def max_context(self):
        try:
            self._max_context = self.cost_tracker.MAX_CONTEXT_WINDOWS[
                self.cost_tracker.model
            ]  # in tokens
        except:
            raise ValueError(f"Model {self.model} not supported")
        return self._max_context

    def build_image_messages(self, image_paths: list[Path]) -> list[dict]:
        """Builds image messages with index numbering for GPT Vision API call."""
        base64_images = [self.encode_image(image_path) for image_path in image_paths]
        messages = []
        for idx, base64_image in enumerate(base64_images, start=1):
            # Add a textual label before each image
            if len(image_paths) > 1:
                messages.append(f"Image {idx}:")
            # Add the image message
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
        return messages

    def encode_image(self, image_path: Path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_token_count(
        self,
        messages: list[dict[str, str]],
        image_paths: Optional[list[Path]],
    ) -> int:
        """Get token count of messages."""
        token_count = 0
        for message in messages:
            if isinstance(message["content"], str):
                token_count += len(self.tokenizer.encode(message["content"]))
            elif isinstance(message["content"], list):
                for msg in message["content"]:
                    if isinstance(msg, str):
                        token_count += len(self.tokenizer.encode(msg))
        if image_paths:
            for image_path in image_paths:
                with Image.open(image_path) as img:
                    w, h = img.size
                token_count += count_image_tokens(width=w, height=h, model=self.model)
        return token_count

    def check_if_exceeds_token_limit(
        self,
        messages: list[dict[str, str]],
        auto_truncate: bool,
        image_paths: Optional[list[Path]] = None,
        num_buffer_tokens: int = 200,
    ) -> list[dict[str, str]]:
        """Ensure total token count of messages does not exceed the maximum context window."""
        token_count = self.get_token_count(messages, image_paths)
        num_images, images_removed = len(image_paths) if image_paths else 0, 0

        if not auto_truncate:
            if token_count > self.max_context:
                raise Exception(
                    f"{token_count} tokens exceed the maximum context window of {self.max_context} for {self.model}."
                )

        while token_count > self.max_context:
            logger.warning(
                f"{token_count} tokens exceed the maximum context window of {self.max_context} for {self.model}. "
                "Truncating user prompt and trying again."
            )
            if not image_paths:
                # Truncate the user prompt text
                user_prompt = messages[1]["content"]
                messages[1]["content"] = self.tokenizer.decode(
                    self.tokenizer.encode(user_prompt)[
                        : (self.max_context - token_count - num_buffer_tokens)
                    ]
                )  # output is truncated if we don't leave a buffer (e.g. buffer of 25 truncates all outputs at 20 tokens)
            else:
                # Truncate the last image
                messages[1]["content"].pop(-1)
                image_paths.pop(-1)
                images_removed += 1

            token_count = self.get_token_count(messages, image_paths)
            logger.warning(f"New token count: {token_count}")

        if images_removed:
            logger.warning(
                f"Removed {images_removed}/{num_images} images to fit within token limit."
            )

        return messages

    def add_to_cache(self, input_hash: str, res: dict[str, str | list]):
        """Insert new API call results into cache."""
        self.cursor.execute(
            "INSERT OR REPLACE INTO gpt_cache (input_hash, response_data) VALUES (?, ?)",
            (input_hash, json.dumps(res)),
        )
        self.cache_count += 1

    def update_cache(self):
        """Commit changes to the database."""
        self.cost_tracker.log_cost()
        logger.info(f"Caching {self.cache_count} API calls, do not force-cancel run...")
        self.conn.commit()
        logger.info("Cache updated.")
        self.cache_count = 0

    def backup_cache(self):
        """Creates or overwrites a single backup of the database using SQLite's backup method."""
        backup_path = self.cache_path.with_suffix(".db.bak")
        # Create or overwrite the backup file
        logger.info(f"Backing up database to {backup_path}...")
        with sqlite3.connect(backup_path) as backup_conn:
            # Perform an efficient backup operation
            self.conn.backup(backup_conn)
        logger.info(f"Database backed up to {backup_path}")

    def load_from_cache(self, input_hash: str):
        """Retrieve response from database cache."""
        self.cursor.execute(
            "SELECT response_data FROM gpt_cache WHERE input_hash = ?", (input_hash,)
        )
        row = self.cursor.fetchone()
        return json.loads(row[0]) if row else None

    def api_input_to_hashable(
        self, messages: list[dict[str, str]], **kwargs
    ) -> tuple[tuple]:
        """Convert API input to hashable string for caching"""
        messages_tuples = [
            (message["role"], message["content"]) for message in messages
        ]
        return str(tuple([self.model] + list(kwargs.items()) + messages_tuples))

    def response_to_dict(
        self,
        response: openai.ChatCompletion,
    ) -> dict[str, str | list]:
        """Converts GPT response to output dict."""
        if isinstance(response, openai.types.chat.chat_completion.ChatCompletion):
            response = response.to_dict()

        return {
            "response": response["choices"][0]["message"]["content"],
            "probs": response_to_tokprobs(response),
            "usage": {
                "completion_tokens": response["usage"]["completion_tokens"],
                "prompt_tokens": response["usage"]["prompt_tokens"],
            },
            "cost": self.cost_tracker.calculate_cost(
                input_tokens=response["usage"]["prompt_tokens"],
                output_tokens=response["usage"]["completion_tokens"],
                model=response["model"],  # defaults to self.model
            ),
        }

    def save_to_batch_file(
        self,
        messages: list[dict[str, str]],
        input_hash: str,
        custom_id: str,
        batch_filepath: Path,
        **kwargs,
    ) -> None:
        """Save messages to batch file for batch API call."""

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                **kwargs,
            },
        }

        custom_id_to_hash = {custom_id: input_hash}

        # File uploaded for batch API call
        with open(batch_filepath, "a") as f:
            f.write(json.dumps(request) + "\n")

        # File mapping custom_ids to hashes for saving to cache
        with open(batch_filepath.with_suffix(".hash"), "a") as f:
            f.write(json.dumps(custom_id_to_hash) + "\n")


class TokProbs(NamedTuple):
    tokens: list[str]
    probs: list[float]


def response_to_tokprobs(
    response: dict,
) -> Optional[list[TokProbs]]:
    """Converts GPT response to list of list of token-logprob tuples."""
    if response["choices"][0]["logprobs"] is None:
        return None

    logprobs = [
        token_logprob["top_logprobs"]
        for token_logprob in response["choices"][0]["logprobs"]["content"]
    ]

    tokprobs = []
    for token_idx in logprobs:
        tokens, probs = [], []
        for token_logprobs in token_idx:
            try:
                tokens.append(token_logprobs.token)
                probs.append(np.exp(token_logprobs.logprob))
            except:
                tokens.append(token_logprobs["token"])
                probs.append(np.exp(token_logprobs["logprob"]))
        tokprobs.append(TokProbs(tokens=tokens, probs=probs))

    return tokprobs
