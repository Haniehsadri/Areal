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

## Offline expert calibration

Use `examples/math/aime_eval_expert.py` for inference-only calibration. The ordinary
GSM8K evaluation script does not export routing. Use your patched vLLM / vLLM
Ascend installation and its existing routing-capture settings: the workflow
requires both expert IDs and gate weights in the response.

The script reads `valid_dataset.path` and `valid_dataset.split`. For expert
selection, set that split to `train`, with `shuffle: false` and `drop_last: false`,
so held-out AIME test questions do not influence selection. Set `gconfig.n_samples`
and generation limits explicitly; this script preserves the config temperature.
The model served by vLLM must be the checkpoint you intend to calibrate.

Run these commands in a Linux shell, replacing the placeholders with your actual
model dimensions and installation details:

```bash
python examples/math/aime_eval_expert.py --config your_calibration_config.yaml \
  --trajectory-output /your/output/aime_trajectories.jsonl \
  --num-moe-layers NUM_MOE_LAYERS \
  --num-experts NUM_EXPERTS \
  --router-top-k ROUTER_TOP_K

python -m areal.tools.select_reward_weighted_experts \
  --trajectories /your/output/aime_trajectories.jsonl \
  --output-dir /your/output/aime_esft_gate \
  --model /your/model --architecture YOUR_MODEL_ARCHITECTURE \
  --backend vllm_ascend --backend-version YOUR_INSTALLED_VERSION \
  --num-moe-layers NUM_MOE_LAYERS --num-experts NUM_EXPERTS \
  --router-top-k ROUTER_TOP_K --experts-per-layer EXPERTS_TO_TRAIN \
  --score-function esft_gate
```

Use `--backend vllm` for CUDA vLLM. Supply the actual tensor/expert parallel sizes
using `--tensor-parallel-size` and `--expert-parallel-size` if they are not one.
Routing dimensions must match the returned tensors; the current CLI assumes
consecutive zero-based MoE layer IDs. Check layer mapping before using a model
with interleaved dense and MoE layers.

The JSONL contains one record per response: prompt/response/request IDs, token
IDs, reward, stop reason, number of scored generation positions, and sparse
per-layer/per-expert gate-weight sums and token-hit counts. Weights come from
vLLM; AReaL accumulates them and counts each selected expert at each scored
position. It uses causal generation alignment, not all prompt tokens.

`esft_gate` averages each response's length-normalized gate scores equally;
`esft_token` uses normalized routing-hit frequencies. `reward_gate` and
`reward_token` weight those response scores by reward. With binary AIME rewards,
the latter use successful responses. The current selector computes all four
scores and rejects a dataset with zero total reward even for plain ESFT.

The selector writes `reward_expert_scores.json` and `esft_config.json` in its
output directory. Repeat selection with another score function and a different
output directory to compare methods using the same captured trajectories.

Use a fresh output filename for each run; the script rejects existing files.
Multi-process/multiple-vLLM-worker capture has not been tested. The workflow's
file lock is local to each instance, so a shared file is not safe across writers.
Use one workflow writer for initial calibration, or separate process-specific
JSONL shards and concatenate completed shards before selection. This entry point
does not automatically assign writer shards. The output path must be writable
and visible to the workflow worker and to the machine running selection.
