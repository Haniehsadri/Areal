# ESFT and FFT optimizer modes

This adaptation restores the previous ESFT implementation's non-distributed
optimizer mode. Expert selection still freezes the model and unfreezes selected
routed experts using global layer/expert names. It retains the current validated
selection JSON and applies freezing before DDP construction so its gradient
buffers cover the trainable parameters.

For ESFT, set these fields in your working config:

```yaml
actor:
  use_esft: true
  esft_config: /your/selection/esft_config.json
  optimizer:
    type: adam
  mindspeed:
    swap_optimizer: false
recover:
  mode: disabled
```

The engine automatically disables distributed optimizer, gradient-reduction
overlap, parameter-gather overlap, and the precision-aware optimizer in its ESFT
configuration. Model/data parallelism remains active. Use Megatron with the
existing `mbridge` integration. Keep your model, hardware, and learning-rate
settings from the working config; this snippet is not a complete launch config.

HF model export remains available. ESFT in this mode cannot save or resume DCP
optimizer/recovery checkpoints because the current checkpoint manager requires a
distributed optimizer. Restarting from an HF export starts fresh optimizer state.

For FFT, set `actor.use_esft: false`. The configured optimizer and DDP settings
remain unchanged, even if the same config originally enabled distributed
optimizer. Use separate experiment/trial names for comparison runs.

This change does not establish the cause of the reported reward collapse. Compare
the same checkpoint, expert selection, prompts, sampling, and optimizer settings;
verify an update and rollout weight synchronization on the target NPU environment.
