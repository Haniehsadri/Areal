# SPDX-License-Identifier: Apache-2.0

"""Prepare local NuminaMath-1.5 cn_k12 data for AReaL RLVR."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any


def convert_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Keep answer-bearing, valid, text-only cn_k12 word problems."""
    if row.get("source") != "cn_k12":
        return None
    if row.get("question_type") != "math-word-problem":
        return None
    if any(row.get(key) != "Yes" for key in ("problem_is_valid", "solution_is_valid")):
        return None
    problem, answer = row.get("problem"), row.get("answer")
    if not isinstance(problem, str) or not isinstance(answer, str):
        return None
    problem, answer = problem.strip(), answer.strip()
    if not problem or not answer or answer.lower() in {"proof", "notfound"}:
        return None
    if re.search(
        r"\[asy\]|!\[|<img|\\includegraphics|\b(?:figure|diagram|pictured)\b",
        problem,
        flags=re.IGNORECASE,
    ):
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", problem).split())
    return {
        "prompt_id": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "messages": [
            {
                "role": "user",
                "content": problem + "\nPlease put your final answer within \\boxed{}.",
            }
        ],
        "answer": answer,
        "source": "cn_k12",
        "problem_type": str(row.get("problem_type") or ""),
    }


def split_records(
    rows: list[dict[str, Any]],
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Deduplicate before splitting; discard prompts with conflicting answers."""
    sizes = (train_size, validation_size, test_size)
    if any(size <= 0 for size in sizes):
        raise ValueError("All split sizes must be positive")
    unique: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for row in rows:
        key = row["prompt_id"]
        if key in unique and unique[key]["answer"] != row["answer"]:
            conflicts.add(key)
        unique.setdefault(key, row)
    eligible = [row for key, row in unique.items() if key not in conflicts]
    if len(eligible) < sum(sizes):
        raise ValueError(
            f"Only {len(eligible)} unique eligible prompts; requested {sum(sizes)}"
        )
    random.Random(seed).shuffle(eligible)
    end_validation = train_size + validation_size
    return {
        "train": eligible[:train_size],
        "validation": eligible[train_size:end_validation],
        "test": eligible[end_validation : end_validation + test_size],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {args.input}")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    files = (
        [args.input]
        if args.input.is_file() and args.input.suffix == ".parquet"
        else sorted(args.input.rglob("*.parquet"))
        if args.input.is_dir()
        else []
    )
    if not files:
        raise ValueError(
            "Input must be a local Parquet file or dataset snapshot directory"
        )

    from datasets import Dataset, DatasetDict, load_dataset

    source = load_dataset(
        "parquet", data_files=[str(file) for file in files], split="train"
    )
    required = {
        "source",
        "question_type",
        "problem_is_valid",
        "solution_is_valid",
        "problem",
        "answer",
    }
    missing = required - set(source.column_names)
    if missing:
        raise ValueError(f"Expected NuminaMath-1.5 columns; missing {sorted(missing)}")
    rows = [converted for row in source if (converted := convert_record(row)) is not None]
    splits = split_records(
        rows, args.train_size, args.validation_size, args.test_size, args.seed
    )
    DatasetDict(
        {name: Dataset.from_list(records) for name, records in splits.items()}
    ).save_to_disk(str(args.output))
    (args.output / "preparation.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "seed": args.seed,
                "source_rows": len(source),
                "eligible_rows_before_deduplication": len(rows),
                "split_sizes": {name: len(records) for name, records in splits.items()},
                "source_filter": "cn_k12",
                "question_type": "math-word-problem",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
