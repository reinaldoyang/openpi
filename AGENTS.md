# TM7S OpenPI project context

This repository contains a custom pi0.5 policy for a Techman TM7S robot. These
instructions apply to the entire repository and are intended to let a new agent
continue the work without reconstructing the training history.

## Safety and authorization

- Never start or resume a training process unless the user explicitly asks for
  that specific run. In particular, do not invoke `scripts/train.py` on your own.
- Never send commands to the physical robot without explicit user approval.
- Develop and validate real-robot integration in prediction-only/dry-run mode
  first. Before actuation, require joint position/velocity/acceleration limits,
  a TCP workspace check, gripper limits, stale-observation handling,
  communication timeouts, and an independently accessible emergency stop.
- Treat empirical training-data ranges as diagnostics, not certified robot
  safety limits.
- Do not commit datasets, checkpoints, W&B files, or evaluation outputs. They
  are excluded by `.gitignore`.

## Current setup: joint space, step-wise delta actions

The project moved from Cartesian TCP actions to joint space on 2026-11-20. The
older TCP-space datasets, checkpoints, and norm stats were deleted from this
server. Do not mix TCP-space and joint-space artifacts, resume from one into the
other, or compare their losses.

There are two joint-space configs in `src/openpi/training/config.py`. They share
the recipe and action convention but use different datasets (see "Dataset"):

- `pi05_tm7s_joint_delta_lora`: 54-episode, 4-task 1119 dataset.
  - Experiment `tm7s_joint_delta_54ep`, trained to step 9999 on 2026-11-20:
    `checkpoints/pi05_tm7s_joint_delta_lora/tm7s_joint_delta_54ep/9999`. Not yet
    evaluated or selected.
  - Norm stats: `assets/pi05_tm7s_joint_delta_lora/local/tm7s_joint_delta_1119/norm_stats.json`.
- `pi05_tm7s_joint_delta_1125_lora`: 50-episode, single-task 1125 dataset.
  - Experiment `tm7s_joint_delta_1125_50ep`, started 2026-11-25 on one GPU:
    `checkpoints/pi05_tm7s_joint_delta_1125_lora/tm7s_joint_delta_1125_50ep/<step>`.
  - Norm stats: `assets/pi05_tm7s_joint_delta_1125_lora/local/tm7s_joint_delta_1125/norm_stats.json`.
- Compute norm stats with
  `uv run scripts/compute_norm_stats.py --config-name <config>` (it runs fine on
  CPU with `JAX_PLATFORMS=cpu`). Recompute them whenever the dataset or
  `exclude_episodes` changes; never reuse stats from another config.
- Training command (only when the user asks for the run; see the next section
  for choosing `<free_gpu>`):

  ```bash
  OPENPI_DATA_HOME=/data/reinaldoyang/openpi_cache \
  HF_DATASETS_CACHE=/data/reinaldoyang/hf_cache/datasets \
  CUDA_VISIBLE_DEVICES=<free_gpu> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py <config> --exp-name=<exp>
  ```

## Training server environment

The training server is shared with other users. Check these before every
training, norm-stats, or serving run on it:

- **Disk.**
  - Keep every large file on `/data` (7 TB). `/` and `/home` are a separate,
    shared 879 GB root disk that is nearly full.
  - openpi's download cache defaults to `~/.cache/openpi` on the root disk.
    Always set `OPENPI_DATA_HOME=/data/reinaldoyang/openpi_cache`. It holds the
    12 GB `pi05_base` weights and the PaliGemma tokenizer.
  - On 2026-11-20 a run without this setting filled `/` to 0 bytes free. The
    run left a truncated `pi05_base` download, which then failed with an Orbax
    `OUT_OF_RANGE ... byte range` error. Orbax still wrote
    `commit_success.txt`, so openpi will not re-download a truncated copy by
    itself; delete it first.
  - LeRobot writes a parquet cache through Hugging Face `datasets`, which
    defaults to `~/.cache/huggingface/datasets` on the root disk. Set
    `HF_DATASETS_CACHE=/data/reinaldoyang/hf_cache/datasets` for norm-stats,
    training, and dataset-loading commands.
  - Run `df -h / /data` before downloading anything large.
- **GPU.**
  - The server has three RTX 6000 Ada GPUs (48 GB each), and other users run
    jobs on them.
  - Run `nvidia-smi` first, pick an idle GPU, and always set
    `CUDA_VISIBLE_DEVICES` to that one GPU. Without it, JAX preallocates
    `XLA_PYTHON_CLIENT_MEM_FRACTION` of memory on every visible GPU and can
    crash other users' jobs.
  - The config uses one GPU (`fsdp_devices=1`).
  - With batch size 3, 10,000 steps took 87-91 minutes on 1, 2, or 3 GPUs
    alike (W&B runs 2026-10-26 to 2026-11-20). Extra GPUs only help with a
    larger global batch, which changes the recipe.
- **Launch.**
  - Start long runs detached
    (`nohup setsid ... > train_<exp>.log 2>&1 < /dev/null &`) so they survive
    the agent session. `*.log` is git-ignored.
  - If a run fails at startup, remove the stub
    `checkpoints/<config>/<exp>/` directory (or pass `--overwrite`) before
    relaunching.
  - W&B credentials are in `~/.netrc`.

- An inference installation needs the complete `params/` tree,
  `_CHECKPOINT_METADATA`, and the checkpoint's
  `assets/local/<repo_id name>/norm_stats.json` (`tm7s_joint_delta_1119` or
  `tm7s_joint_delta_1125`). `train_state/` is not
  required. Checkpoints use JAX/Orbax; preserve every file under `params/`,
  because individual OCDBT files are not independently usable.

## Model configuration

- pi0.5 with a 16-action horizon, a 32-dimensional internal action space, and
  `max_token_len=80`. The longest training prompt was measured at 64 tokens.
- Both the Gemma 2B branch and 300M action expert use LoRA (`gemma_2b_lora`,
  `gemma_300m_lora`), initialized from `pi05_base`. Their original weights are
  frozen, but this is not strictly adapter-only: the full SigLIP vision encoder
  and the small action/time projections are trainable (about 467M of 3.40B
  parameters in the same recipe on the earlier TCP config).
- `ema_decay=None`, 10,000 steps, global batch size 3, `fsdp_devices=1`, W&B
  logging.
- `LeRobotTM7SDataConfig` flags for this config, which must stay as set:
  - `use_delta_pose_actions=False`: the dataset already stores deltas, so no
    `DeltaActions`/`AbsoluteActions` transform is applied.
  - `wrap_rx=False`: `WrapAngles` maps dim 3 into `[0, 2*pi)`, which was meant
    for TCP `rx`. In joint space dim 3 is joint 4 (about -34.5 to 42.1 deg), so
    wrapping would corrupt it. The data input transforms must be only
    `LiberoInputs`.

## State and action convention

Both are 7 values: six joints in radians, then gripper.

- `observation.state`: measured `joint_1..joint_6` (rad) plus the measured
  gripper as a continuous value `(85 - width_mm) / 85`
  (`0.0 = open = 85 mm`, `1.0 = closed`).
- `action`: `commanded_joints_t - measured_joints_t` (rad) for joints 1-6, plus
  a binary absolute gripper command (0 = open, 1 = closed).
  - The delta is step-wise: each step is relative to the measured joints at the
    same step, not to the state at the start of the chunk.
  - Recorded deltas are at most about 0.04 rad (2.3 deg) per step in the 1119
    data and 0.053 rad (3.05 deg) in the faster 1125 data.
- Because no output delta transform is applied, the policy returns a `(16, 7)`
  chunk of joint deltas plus gripper, not absolute joint targets. To execute
  step `k`, command `measured_joints_now + actions[k, :6]`, reading the
  measured joints fresh at that step. TMRTC joint mode uses degrees.
- The gripper output is continuous near 0 or 1. The controller must clamp and
  discretize it, preferably with hysteresis (for example close at `>= 0.7`,
  open at `<= 0.3`, otherwise keep the current state).
- The live state, units, joint ordering, camera orientation/cropping, and
  gripper semantics must exactly match training.

## Camera and observation pipeline

The dataset contains two independent 640x480 camera streams, not stitched into
one image.

- `observation.images.table` becomes `observation/image`, then model input
  `base_0_rgb`.
- `observation.images.wrist` becomes `observation/wrist_image`, then model input
  `left_wrist_0_rgb`.
- A black `right_wrist_0_rgb` image is created and masked out.
- Each real image is independently converted to HWC `uint8`, resized/padded to
  224x224, and encoded by the shared SigLIP vision encoder.

Runtime observation schema:

```python
observation = {
    "observation/state": joints_rad_and_gripper,  # float, shape (7,)
    "observation/image": table_rgb,               # uint8 HWC
    "observation/wrist_image": wrist_rgb,         # uint8 HWC
    "prompt": task,                               # one of the task strings below
}
```

## Dataset

Two joint-space datasets, both LeRobot v3.0 at 15 Hz (the 16-step horizon
covers about 1.07 seconds):

- 1119: `data/tm7s_lerobot_joint_54ep_1119`, `repo_id`
  `local/tm7s_joint_delta_1119` (collected 2026-11-19). 54 episodes, 32,994
  frames, the 4 tasks below.
  - Converted with the `tm7s_teleop_rtrs` converter (`--action-space joint`,
    commit `bae2ce3`). The folder's `HANDOFF.md` describes the conversion.
  - No longer on this server (as of 2026-11-25); its norm stats and checkpoint
    remain.
- 1125: `data/tm7s_lerobot_joint_50ep_1125`, `repo_id`
  `local/tm7s_joint_delta_1125` (collected 2026-11-25). 50 episodes, 11,119
  frames (170-291 per episode), single task
  `Pick up the bowl and place it in the green basket`.
  - Converted with `teleoperation.convert_tm7s_to_lerobot` (LeRobot 0.4.4,
    `action_space: joint`); see `tm7s_conversion_manifest.json` in the folder.
  - Verified on all frames: same state/action convention as 1119, gripper state
    `(85 - width_mm) / 85`, gripper commands only 0/85 mm.
  - Recorded faster than 1119: measured joint speed up to about 25 deg/s.
  - A first copy was taken mid-conversion (truncated parquet files, 38
    episodes). If a dataset will not open, check that conversion finished.
- Both cameras use AV1 video, decoded with the PyAV backend.
- `auxiliary.*` columns (raw joints, gripper widths in mm) are not model inputs.
- The prompt is the per-episode `task`. Use these exact strings at inference
  (1119 tasks; the 1125 task is the last one):
  - 24 episodes: `Pick up all the bowls and place it in the green basket and pick up all the cups and place it in the red basket`
  - 10 episodes: `Pick up all the bowl and place it in the green basket`
  - 10 episodes: `Pick up all the cups and place it in the red basket`
  - 10 episodes: `Pick up the bowl and place it in the green basket`
- Dependency compatibility requires the repository's `pyarrow<26` override;
  use the committed `uv.lock`.

## Tools that still assume TCP space

These were written for the earlier Cartesian models and are not compatible with
the joint-delta policy as they stand:

- `examples/tm7s/evaluate_offline.py` reports TCP position error in metres from
  action dims 0-2. Joint-space evaluation needs per-joint error in radians
  (optionally TCP error through forward kinematics). It must never be
  repurposed to send robot commands.
- `examples/tm7s/rollout.py` streams Cartesian TCP targets. See the next
  section for what must change.

Build or adapt either only when the user asks. A rollout client must start in
prediction-only mode and follow the safety rules above. Ask the user for the
confirmed safety limits before implementing command execution.

## Adapting `examples/tm7s/rollout.py` to joint space

Ask the user whether to add a joint mode to `rollout.py` or write a separate
script. Keep the TCP path working, because the archived TCP checkpoint still
depends on it.

### Hardware dependencies

`rollout.py` imports its hardware primitives from the `tm7s_keyboard_teleop`
repository (`--teleop-root`, default `/home/reinaldoyang/tm7s_keyboard_teleop`).
The modules are `teleoperation.tmrtc_cartesian_xyz_teleop`,
`teleoperation.tmrtc_data_collection` and
`testing.tmrtc_cartesian_hold_test`. The joint dataset was collected and
converted with `tm7s_teleop_rtrs` (converter commit `bae2ce3`). Neither repo is
on the training server.

On the lab machine, find the joint-mode TMRTC primitives the joint collector
used (mode settings packet, joint target packet, units) and reuse them. Do not
write new protocol code.

### Code that assumes TCP space

The following were verified against the current file:

- **`ObservationSource.capture`**
  - State is built from `feedback.state.base_pose` (mm/deg converted to
    m/rad). Use the measured joints instead (`feedback.state.joints`; check its
    units in the teleop repo; the model needs radians).
  - The gripper state uses `(85 - width) / (85 - 6.5)`. The joint dataset uses
    `(85 - width_mm) / 85` (verified on all 32,994 frames). The old formula
    differs by up to 0.076, so change the span to 85.
- **`CommandFilter`**
  - It converts an absolute TCP pose to mm/deg, clips to the RobotBase
    workspace, locks orientation (`lock_orientation=True`), and rate-limits
    translation and orientation.
  - Replace it with a joint filter: target `measured_joints_now +
    actions[k, :6]`, converted rad to deg, with a fresh, non-stale feedback
    read every tick.
  - Add per-joint position, speed and acceleration limits, and a TCP workspace
    check through forward kinematics.
- **`RolloutRuntime._build_cartesian_setting` / `_start_live_hold` / `_tick`**
  - They send the Cartesian mode settings (`,8,`), build the filter from
    `state.base_pose`, and send `build_tmrtc_pose_target(pose)`. All of these
    need joint-mode equivalents.
- **`RolloutRuntime._request_inference_if_needed` and the loop in `run`**
  - The settle check before inference uses `INFERENCE_SETTLED_POSITION_MM` /
    `INFERENCE_SETTLED_ORIENTATION_DEG`.
  - Live tracking aborts at `TRACKING_ABORT_MM` /
    `MAX_TRACKING_ORIENTATION_ERROR_DEG`.
  - Both use the Cartesian `pose_error`. Use joint-space errors or forward
    kinematics instead.
- **Gripper width**
  - `RolloutGripper` commands 6.5 mm on CLOSE. During joint data collection,
    the commanded widths were exactly 0 mm (closed) and 85 mm (open).
  - Decision (user, 2026-11-20): joint mode commands exactly 0 mm on CLOSE and
    85 mm on OPEN, matching collection.
  - `GRIPPER_CLOSED_WIDTH_MM = 6.5` is also the TCP model's gripper-state
    span, so do not edit it. Give joint mode its own close width (0 mm) and
    state span (85 mm).
  - `_apply_gripper` hysteresis (close `>= 0.70`, open `<= 0.30`) can stay.

### Facts for limit design

These are diagnostics, not safety limits:

- Recorded action deltas peak at 2.34, 2.29, 2.33, 2.25, 2.15, 1.61 deg for
  joints 1-6.
- Each delta is commanded minus measured at the same step: the controller's
  tracking offset, not per-step motion. Do not treat it as a velocity.
- Data collection limited joint speed to 17 deg/s. Reject or stop on any
  predicted delta well outside the recorded range instead of clipping silently.
- The 1125 dataset was recorded faster: action deltas peak at 3.05, 2.55, 3.04,
  2.85, 2.48, 1.93 deg and measured joint speed at about 25 deg/s. Its
  collection speed limit is unconfirmed. Use the ranges of the dataset the
  checkpoint was trained on.

### Keep unchanged

- Prediction-only default, `--enable-actuation` plus Enter confirmation, and
  `--hold-only`.
- Stale feedback/camera checks and camera skew check.
- `MAX_JOINT_CHANGE_PER_SAMPLE_DEG` feedback check.
- Inference timeout and stale-result rejection.
- Generation invalidation (X), software stop, and guarded joint home.

### Validation order

1. Prediction-only against the policy server, logging predicted joint targets.
2. Compare those targets with the dataset ranges.
3. `--hold-only` stop test.
4. Live actuation, only with explicit user approval each session.

## Local inference

```bash
uv sync
uv lock --check
uv run python -c "import jax; print(jax.devices())"
# On the shared training server, also set OPENPI_DATA_HOME and
# CUDA_VISIBLE_DEVICES as described in "Training server environment".
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_tm7s_joint_delta_lora \
  --policy.dir=checkpoints/pi05_tm7s_joint_delta_lora/tm7s_joint_delta_54ep/<step> \
  --port=8000
```

If the PaliGemma tokenizer cannot be downloaded automatically, place
`paligemma_tokenizer.model` at
`$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`.

## Earlier TCP-space work (archived)

- `pi05_tm7s_finetune` (50-episode bowl task) and
  `pi05_tm7s_multitask_finetune` (102-episode multitask) remain in
  `config.py`, but their local datasets, checkpoints, and norm stats were
  deleted.
- The TCP bowl checkpoint (step 9999) is still in the private Hugging Face repo
  `Reinaldoyang/pi05-tm7s-bowl-green-basket`. It uses absolute TCP pose actions
  and is not interchangeable with the joint-space policy.
