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
        (r"<think>\boxed{42}</think>\boxed{43}", "42", 0.0),
        (r"<think>\boxed{43}</think>\boxed{42}", "42", 1.0),
        (r"reasoning \boxed{42}</think>No answer.", "42", 0.0),
        (r"<think>\boxed{42}", "42", 1.0),
        ("", "42", 0.0),
    ],
)
def test_aime_final_answer_scoring(response, gold, expected):
    """Preserve the supplied verifier's answer-extraction behavior."""
    worker = ThinkingMathVerifyWorker()
    assert worker.verify(response, gold) == expected


def test_aime_missing_gold_returns_zero():
    """Missing labels cannot produce a positive reward."""
    assert aime_reward_fn("", r"\boxed{42}", [], [], None) == 0.0


@pytest.mark.parametrize(
    "response,expected",
    [
        ("first</think>second</think>third", "second</think>third"),
        (r"<think>\boxed{42}", r"<think>\boxed{42}"),
    ],
)
def test_aime_original_preprocessing_preserved(response, expected):
    """Use the first closing tag and always box gold exactly as supplied."""
    worker = ThinkingMathVerifyWorker()
    calls = []

    def metric(gold, prediction):
        calls.append((gold, prediction))
        return 0.75, None

    worker.verify_func = metric
    assert worker.verify(response, r"\boxed{42}") == 0.75
    assert calls == [([r"\boxed{\boxed{42}}"], [expected])]
