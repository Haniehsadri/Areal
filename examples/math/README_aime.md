# AIME adaptation

The adapted files are `areal/reward/__init__.py`, `areal/reward/aime.py`,
and `examples/math/aime_rl.py`. The shared reward module retains the current
parser API; do not replace it with the old copy. The old `KDRLWorkflow` is
absent in this checkout, so this entry point rejects teacher distillation.

The existing local loader expects a directory whose path contains `aime`,
with both `aime_train.parquet` and `aime_test.parquet`. Each file needs
`question` and `answer` columns. Answers are final answers, not worked solutions.
An arbitrary downloaded AIME file may need conversion to this schema first.

Copy your working hardware/model training config and update these sections:

```yaml
train_dataset:
  path: ./aime_data
  type: rl
  split: train
  datasets: []
  dataset_kwargs: {}
valid_dataset:
  path: ./aime_data
  type: rl
  split: test
  datasets: []
  dataset_kwargs: {}
  shuffle: false
  drop_last: false
```

Replace both paths with your actual local directory. Both files must exist
because the current loader opens both splits. Use independent train and test
questions; do not copy the same questions into both files.

```bash
python examples/math/aime_rl.py --config your_config.yaml
```

This launches training with validation, not standalone evaluation. It honors
the configured splits, enables thinking in the chat template, and retains
your friend's evaluation temperature of 0.6. The reward excludes text before
the final `</think>` and verifies the remaining answer. An explicitly opened
but unfinished thinking block scores zero. If generation omits both thinking
tags, the whole completion is checked. Plain gold answers are boxed for parsing.

Keep your working generation limits initially and check completion truncation
before interpreting accuracy. Thinking mode depends on the model's chat template.
This script does not add expert-routing capture.

Run the scoring tests in your AReaL environment:

```bash
python -m pytest tests/test_aime_reward.py
```
