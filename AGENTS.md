# TM7S OpenPI project context

This repository contains a custom pi0.5 policy for a Techman TM7S robot. These
instructions apply to the entire repository and are intended to let a new agent
continue the work without reconstructing the training history.

## Safety and authorization

- Never start or resume a training process unless the user explicitly asks for
  that specific run. In particular, do not invoke `scripts/train.py` on your own.
- Never send commands to the physical robot without explicit user approval.
- Develop and validate real-robot integration in prediction-only/dry-run mode
  first. Before actuation, require workspace, velocity, acceleration, and
  gripper limits; stale-observation handling; communication timeouts; and an
  independently accessible emergency stop.
- Treat empirical training-data ranges as diagnostics, not certified robot
  safety limits.
- Do not commit datasets, checkpoints, W&B files, or evaluation outputs. They
  are excluded by `.gitignore`.

## Current model and checkpoint

- Training config: `pi05_tm7s_finetune` in `src/openpi/training/config.py`.
- Selected checkpoint: step `9999` from experiment
  `tm7s_bowl_green_basket_lora`.
- Private Hugging Face model repository (namespace is case-sensitive):
  `Reinaldoyang/pi05-tm7s-bowl-green-basket`.
- Expected local checkpoint path:
  `checkpoints/pi05_tm7s_finetune/tm7s_bowl_green_basket_lora/9999`.
- An inference installation needs the complete `params/` tree,
  `_CHECKPOINT_METADATA`, and
  `assets/local/tm7s_bowl_green_basket/norm_stats.json`.
- `train_state/` is not required for inference. The original pi0.5 base
  checkpoint and the training dataset are also not required because the saved
  `params/` tree contains the complete model.

Download the private checkpoint with an authenticated Hugging Face account:

```bash
uv run hf auth login
uv run hf download Reinaldoyang/pi05-tm7s-bowl-green-basket \
  --local-dir checkpoints/pi05_tm7s_finetune/tm7s_bowl_green_basket_lora/9999
```

The checkpoint uses JAX/Orbax. Preserve every file and subdirectory under
`params/`; individual OCDBT files are not independently usable.

## Model configuration

- pi0.5 with a 16-action horizon and a 32-dimensional internal action space.
- The robot state/action space has 7 values:
  `[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad, gripper]`.
- The first six training action dimensions are pose deltas from the current TCP
  state. The gripper remains absolute. Inference transforms the first six back
  to absolute TCP commands, so the returned `(16, 7)` action chunk is in the
  robot's original absolute convention.
- The gripper dataset convention is `0.0 = open = 85.0 mm` and
  `1.0 = closed = 6.5 mm`. The model returns a continuous value near zero or
  one; the controller must clamp and discretize it. Prefer hysteresis (for
  example close at `>= 0.7`, open at `<= 0.3`, otherwise retain state) over an
  unfiltered threshold if command chatter is possible.
- `max_token_len=64`; the current prompt/state encoding was measured at no more
  than 50 tokens.
- Both the Gemma 2B branch and 300M action expert use LoRA. Their original
  language/action-transformer weights are frozen, but this is not a strictly
  adapter-only run: the full SigLIP vision encoder and the small action/time
  projections were trainable. Approximately 467M of 3.40B parameters were
  trainable (414.8M vision, 50.0M LoRA, and about 2.2M projections/time MLPs).
- The config uses `ema_decay=None`, global batch size 3, `fsdp_devices=1`, and
  W&B logging.

## Camera and observation pipeline

The training dataset contains two independent 640x480 camera streams. They are
not stitched into one image.

- `observation.images.table` becomes `observation/image`, then model input
  `base_0_rgb`.
- `observation.images.wrist` becomes `observation/wrist_image`, then model input
  `left_wrist_0_rgb`.
- A black `right_wrist_0_rgb` image is created and masked out.
- Each real image is independently converted to HWC `uint8`, resized/padded to
  224x224, and encoded by the shared SigLIP vision encoder. The resulting image
  tokens are concatenated with the prompt/state tokens; pixels are not combined.

Use this runtime observation schema:

```python
observation = {
    "observation/state": tcp_pose_and_gripper,  # float, shape (7,)
    "observation/image": table_rgb,             # uint8 HWC
    "observation/wrist_image": wrist_rgb,       # uint8 HWC
    "prompt": "Pick up the bowl and place it in the green basket",
}
```

The live state, units, coordinate frame, rotation-vector convention, camera
orientation/cropping, and gripper semantics must exactly match training.

## Dataset history

- Original server-relative path:
  `data/real_50ep_1026/real_robot_data/tm7s_lerobot`.
- LeRobot v3 dataset, 50 episodes, 11,698 frames, 15 Hz.
- Both cameras use AV1 video and were decoded successfully with the PyAV
  backend.
- Action horizon 16 covers about 1.07 seconds at 15 Hz.
- Dependency compatibility requires the repository's `pyarrow<26` override;
  use the committed `uv.lock`.

## Verified offline evaluation

The evaluator is `examples/tm7s/evaluate_offline.py`. It loads checkpoint
normalization assets, decodes both cameras, uses deterministic frame selection
and diffusion noise, excludes padded episode-tail actions, and writes JSON/NPZ
under the ignored `eval_results/` directory.

Checkpoint 9999, 100 sampled training frames (44 of 50 episodes, 1,559 valid
actions):

- First action position error: mean 1.571 mm, p95 3.491 mm, max 4.655 mm.
- Full chunk position error: mean 5.685 mm, p95 19.807 mm, max 56.234 mm.
- Horizon-step-15 mean position error: 9.783 mm.
- First-action gripper accuracy: 96.0%.
- Full-chunk gripper accuracy: 98.076% (30 mistakes).
- Steady-state inference time: about 1.139 seconds per chunk on the training
  server.

Checkpoint 9999 was better than checkpoint 5000 on the identical inputs at
every horizon step, so 9999 is the selected checkpoint. These metrics measure
training-data imitation, not generalization or task success. Orientation barely
varies in the dataset, so its very small offline error is not a strong result.
Aggregate gripper accuracy also hides transition timing: among 17 sampled chunks
with a target transition, only 64.7% were correct at the transition action, 2/17
matched the exact transition index, and 7/17 were within one 15 Hz frame.

## Local inference and real-robot rollout

First verify dependencies, GPU access, and checkpoint loading:

```bash
uv sync
uv lock --check
uv run python -c "import jax; print(jax.devices())"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_tm7s_finetune \
  --policy.dir=checkpoints/pi05_tm7s_finetune/tm7s_bowl_green_basket_lora/9999 \
  --port=8000
```

If the PaliGemma tokenizer cannot be downloaded automatically, place
`paligemma_tokenizer.model` at
`$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`.

The TM7S real-robot rollout client has not yet been implemented in this
repository. `examples/tm7s/evaluate_offline.py` must not be repurposed to send
robot commands. A rollout client should use `openpi_client` (locally via
`localhost:8000`, or over a trusted LAN to a GPU server), acquire synchronized
camera/state observations, enforce safety constraints, and initially log
predictions without actuation. Ask the user for the TM controller API, camera
capture interfaces, control frequency, and confirmed safety limits before
implementing command execution.

