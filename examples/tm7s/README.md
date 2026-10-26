# Fine-tuning pi0.5 on the TM7S dataset

The `pi05_tm7s_finetune` config is set up for the dataset at:

```text
data/real_50ep_1026/real_robot_data/tm7s_lerobot
```

The dataset was produced with LeRobot 0.4.4 in the v3.0 format. This repository
therefore pins LeRobot 0.4.4 and passes the local dataset root to LeRobot instead
of trying to download `local/tm7s_bowl_green_basket` from the Hub.
The config explicitly uses the PyAV video backend, which was verified to decode
both AV1 camera streams without requiring TorchCodec-compatible system FFmpeg
libraries.

## Data mapping

The pipeline uses the table camera as `base_0_rgb`, the wrist camera as
`left_wrist_0_rgb`, and a masked black image for the unavailable right wrist
camera. The seven state/action values are:

```text
[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad, gripper]
```

The source actions are absolute TCP commands. During training, the first six
pose dimensions are converted to deltas from the current measured TCP state;
the gripper remains absolute. Inference converts the predicted pose deltas back
to absolute TCP commands.

This is a robot/action-space choice, not a requirement of pi0.5. Set
`use_delta_pose_actions=False` in `LeRobotTM7SDataConfig` to train directly on
absolute TCP targets. If this setting changes, recompute normalization stats.

The action horizon is 16 frames, or about 1.07 seconds at 15 Hz. Actions and
state are padded from 7 to pi0.5's 32-dimensional model space after the robot
transforms run.

The fine-tuning config uses LoRA adapters for both the PaliGemma backbone and
the action expert. Base language-model weights are frozen, EMA is disabled, and
the model is replicated across the server's three GPUs (`fsdp_devices=1`) for
data-parallel training with a global batch size of 3. The 64-token input limit
was validated against all 11,698 dataset frames; the longest encoded prompt and
state uses 50 tokens. Increase this limit if inference will use longer prompts.

## Server commands

Run commands from the repository root. If the dataset is elsewhere on the
server, change `root` in `pi05_tm7s_finetune` in
`src/openpi/training/config.py` first.

```bash
uv sync
uv run scripts/compute_norm_stats.py --config-name pi05_tm7s_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_tm7s_finetune --exp-name=tm7s_bowl_green_basket --overwrite
```

Do not reuse base-model normalization statistics for this robot. The stats must
be computed after the absolute-to-delta action transform, as the command above
does.

For inference, provide observations using these runtime keys:

```python
observation = {
    "observation/state": tcp_pose_and_gripper,  # float array, shape (7,)
    "observation/image": table_rgb,             # uint8 HWC
    "observation/wrist_image": wrist_rgb,       # uint8 HWC
    "prompt": "Pick up the bowl and place it in the green basket",
}
```

The policy returns `actions` with shape `(16, 7)` in the dataset's absolute TCP
command convention. Apply robot-side safety limits before sending predictions
to the TM controller.

## Offline training-data evaluation

The offline evaluator decodes observations from the local training dataset,
runs the policy without connecting to the robot, and compares its predicted
16-step action chunks with the recorded chunks. Padded actions at episode ends
are excluded. It reports position error in meters, SO(3) geodesic orientation
error, gripper accuracy, per-horizon metrics, and predictions outside the
empirical training-data range.

Evaluate the final LoRA checkpoint on 100 deterministic random frames:

```bash
uv run examples/tm7s/evaluate_offline.py
```

The summary JSON and optional raw prediction NPZ are written under
`eval_results/tm7s_offline/`. To compare the retained 5,000-step checkpoint on
the exact same observations and diffusion noise, keep `--num-samples` and
`--seed` unchanged:

```bash
uv run examples/tm7s/evaluate_offline.py \
  --checkpoint-dir checkpoints/pi05_tm7s_finetune/tm7s_bowl_green_basket_lora/5000
```

For a quick smoke test, use `--num-samples 2`. This evaluation measures how
well the policy imitates recorded training actions. It does not measure task
success, generalization, or closed-loop robot safety; those require held-out
episodes and carefully supervised robot rollouts.
