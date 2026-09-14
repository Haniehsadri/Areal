# SPDX-License-Identifier: Apache-2.0

from typing import Any

from areal.utils import logging

from . import get_thinking_math_verify_worker

logger = logging.getLogger("RewardUtils")


def aime_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    answer: str | int | None = None,
    **kwargs: Any,
) -> float:
    """Score the final mathematical answer, excluding completed thinking blocks."""
    try:
        return get_thinking_math_verify_worker().verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in aime_reward_fn", exc_info=True)
        return 0.0
