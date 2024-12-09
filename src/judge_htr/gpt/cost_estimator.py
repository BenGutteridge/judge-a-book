import functools
from loguru import logger
from math import ceil


class CostTracker:
    """Mostly from [0].
    [0]: https://github.com/michaelachmann/gpt-cost-estimator
    """

    MODEL_SYNONYMS = {
        "gpt-4": "gpt-4-0613",
        "gpt-3-turbo": "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    }

    # Source: https://openai.com/pricing
    # Prices in $ per 1000 tokens
    # Last updated: 2024-01-26
    PRICES = {
        "gpt-4-0613": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-vision-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        # "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k-0613": {"input": 0.003, "output": 0.004},
        "whisper-1": {"input": 0.006, "output": 0.006},
        "tts-1": {"input": 0.015, "output": 0.015},
        "tts-hd-1": {"input": 0.03, "output": 0.03},
        "text-embedding-ada-002-v2": {"input": 0.0001, "output": 0.0001},
        "text-davinci:003": {"input": 0.02, "output": 0.02},
        "text-ada-001": {"input": 0.0004, "output": 0.0004},
        # Added/confirmed 2024-07
        # "gpt-4o": {"input": 2.5e-3, "output": 10e-3},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.15e-3, "output": 0.60e-3},
        "gpt-4o-mini-2024-07-18": {"input": 0.15e-3, "output": 0.60e-3},
        # Added fine-tuning info 2024-08-28
        "ft:gpt-4o-mini-2024-07-18": {"input": 0.3e-3, "output": 1.2e-3},
        "ft:gpt-4o-2024-08-06": {"input": 3.75e-3, "output": 15e-3},
        # Added 2024-11-30
        "gpt-4o": {"input": 2.5e-3, "output": 10e-3},
        "gpt-4o-2024-08-06": {"input": 2.5e-3, "output": 10e-3},
    }

    MAX_CONTEXT_WINDOWS = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-3.5-turbo": 16_385,
        "gpt-4-turbo": 128_000,
        # Fine-tuned models are formatted as ft:<model>:<organization>:<ft-job-id>
        # We remove the org/job info in `caller.py` and just keep ft:<model>
        "ft:gpt-4o-2024-08-06": 128_000,
        "ft:gpt-4o-mini-2024-07-18": 128_000,
    }

    total_cost = 0.0  # class variable to persist total_cost

    def __init__(self, model: str, limit_in_usd: int = 50) -> None:
        self.model = model
        self.limit_in_usd = limit_in_usd

    @classmethod
    def reset(cls) -> None:
        cls.total_cost = 0.0

    def __call__(self, function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            response = function(*args, **kwargs)
            input_tokens = response.usage.prompt_tokens
            if input_tokens > self.MAX_CONTEXT_WINDOWS[self.model]:
                logger.warning(
                    f"{input_tokens} input tokens exceed the maximum context window of {self.MAX_CONTEXT_WINDOWS[self.model]} for {self.model}."
                )
            output_tokens = response.usage.completion_tokens
            cost = self.calculate_cost(input_tokens, output_tokens)
            self.total_cost += cost
            return response

        return wrapper

    def log_cost(self):
        logger.log("COST", f"Running cost: ${self.total_cost}")
        if self.total_cost > self.limit_in_usd:
            raise Exception(f"Cost exceeded ${self.limit_in_usd}.")

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str = None
    ) -> float:
        model = model or self.model  # allow overriding model
        input_cost = input_tokens * self.PRICES[model]["input"] / 1000
        output_cost = output_tokens * self.PRICES[model]["output"] / 1000
        return input_cost + output_cost


def count_image_tokens(
    width: int,
    height: int,
    model: str,
    detail: str = "high",
) -> int:
    """Calculate approximate token cost for an image.
    Written by GPT-4o fed these instructions:
    https://platform.openai.com/docs/guides/vision/calculating-costs
    Edited to include GPT-4o-mini differences.
    """
    assert detail in ["low", "high"], "detail must be 'low' or 'high'"
    assert any(m in model for m in ["gpt-4o", "gpt-4o-mini"])  # vision-capable models

    # Vision processing costs the same for both models
    # As of 2024-08-08, 4o-mini is ~33x cheaper than 4o per token, and inflates image tokens by ~33x
    # https://community.openai.com/t/gpt-4o-mini-high-vision-cost/872382/6
    base_tokens = 85 if "mini" not in model else 2833
    per_tile_tokens = 170 if "mini" not in model else 5667

    if detail == "low":
        return base_tokens

    # Scale down the image to fit within the 2048 x 2048 square.
    if max(width, height) > 2048:
        ratio = 2048.0 / max(width, height)
        width, height = round(width * ratio), round(height * ratio)

    # Scale down such that the shortest side of the image is 768px long.
    ratio = 768.0 / min(width, height)
    width, height = round(width * ratio), round(height * ratio)

    # Count how many 512px squares the image consists of.
    tiles = ceil(width / 512) * ceil(height / 512)

    # Each tile costs 170 tokens, and add 85 to the final total.
    token_count = tiles * per_tile_tokens + base_tokens

    return token_count


FINETUNING_COSTS = {  # $ per 1M training tokens
    "ft:gpt-4o-2024-08-06": 25,
    "ft:gpt-4o-mini-2024-07-18": 3,
}
