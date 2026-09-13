# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.reward import ThinkingMathVerifyWorker
from areal.reward.aime import aime_reward_fn


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        (r"\boxed{42}", "42", 1.0),
        (r"\boxed{43}", "42", 0.0),
        (r"\boxed{42}", "042", 1.0),
        (r"\boxed{42}", r"\boxed{42}", 1.0),
        (r"<think>\boxed{42}</think>\boxed{43}", "42", 0.0),
        (r"<think>\boxed{43}</think>\boxed{42}", "42", 1.0),
        (r"reasoning \boxed{42}</think>No answer.", "42", 0.0),
        (r"<think>\boxed{42}", "42", 0.0),
        ("", "42", 0.0),
    ],
)
def test_aime_final_answer_scoring(response, gold, expected):
    """Reasoning answers must not override the final answer."""
    worker = ThinkingMathVerifyWorker(timeout=None)
    assert worker.verify(response, gold) == expected


def test_aime_missing_gold_returns_zero():
    """Missing labels cannot produce a positive reward."""
    assert aime_reward_fn("", r"\boxed{42}", [], [], None) == 0.0
