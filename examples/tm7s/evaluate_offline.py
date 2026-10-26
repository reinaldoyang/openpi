"""Evaluate a trained TM7S policy on observations from its LeRobot dataset.

This is an offline imitation metric: it compares predicted action chunks with
recorded action chunks. It never connects to or commands the robot.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import pathlib
import time
from typing import Any

import jax
import numpy as np
import tqdm
import tyro

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader

_ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")


@dataclasses.dataclass(frozen=True)
class Args:
    """Command-line arguments for TM7S offline evaluation."""

    config_name: str = "pi05_tm7s_finetune"
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_tm7s_finetune/tm7s_bowl_green_basket_lora/9999")
    num_samples: int = 100
    seed: int = 42
    num_inference_steps: int = 10
    gripper_threshold: float = 0.5
    range_margin_fraction: float = 0.0
    output_dir: pathlib.Path = pathlib.Path("eval_results/tm7s_offline")
    save_predictions: bool = True


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/JAX/NumPy values to host NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation vectors to xyzw unit quaternions."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = angle / 2.0
    scale = np.divide(
        np.sin(half_angle),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1e-12,
    )
    xyz = rotvec * scale
    return np.concatenate([xyz, np.cos(half_angle)], axis=-1)


def rotation_geodesic_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) angle between rotation vectors, in radians."""
    predicted_quat = _rotvec_to_quaternion(predicted)
    target_quat = _rotvec_to_quaternion(target)
    # q and -q encode the same rotation, hence abs(dot).
    cosine_half_angle = np.abs(np.sum(predicted_quat * target_quat, axis=-1))
    return 2.0 * np.arccos(np.clip(cosine_half_angle, 0.0, 1.0))


def _scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _action_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    gripper_threshold: float,
) -> dict[str, Any]:
    position_error = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1)
    orientation_error = rotation_geodesic_error(predicted[..., 3:6], target[..., 3:6])
    gripper_error = np.abs(predicted[..., 6] - target[..., 6])

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            raise ValueError("Metric mask has no valid actions")
        pred_binary = predicted[..., 6] >= gripper_threshold
        target_binary = target[..., 6] >= gripper_threshold
        return {
            "count": int(mask.sum()),
            "position_l2_m": _scalar_summary(position_error[mask]),
            "position_axis_mae_m": {
                name: float(np.mean(np.abs(predicted[..., i] - target[..., i])[mask]))
                for i, name in enumerate(_ACTION_NAMES[:3])
            },
            "orientation_geodesic_rad": _scalar_summary(orientation_error[mask]),
            "orientation_geodesic_deg_mean": float(np.rad2deg(np.mean(orientation_error[mask]))),
            "gripper_mae": float(np.mean(gripper_error[mask])),
            "gripper_binary_accuracy": float(np.mean(pred_binary[mask] == target_binary[mask])),
        }

    horizon = []
    for step in range(predicted.shape[1]):
        step_mask = valid[:, step]
        count = int(step_mask.sum())
        horizon.append(
            {
                "step": step,
                "count": count,
                "position_l2_m_mean": float(np.mean(position_error[:, step][step_mask])) if count else None,
                "orientation_geodesic_rad_mean": (
                    float(np.mean(orientation_error[:, step][step_mask])) if count else None
                ),
                "gripper_mae": float(np.mean(gripper_error[:, step][step_mask])) if count else None,
            }
        )

    return {
        "all_valid_actions": summarize(valid),
        "first_action": summarize(valid & (np.arange(predicted.shape[1])[None, :] == 0)),
        "by_horizon_step": horizon,
    }


def _training_range_metrics(
    predicted: np.ndarray,
    valid: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
    margin_fraction: float,
) -> dict[str, Any]:
    span = action_max - action_min
    lower = action_min - margin_fraction * span
    upper = action_max + margin_fraction * span
    outside = (predicted < lower) | (predicted > upper)
    valid_dims = np.broadcast_to(valid[..., None], predicted.shape)
    valid_outside = outside & valid_dims
    valid_action_outside = np.any(outside, axis=-1) & valid
    denominator = int(valid.sum())
    return {
        "description": "Predictions outside the empirical min/max action range in the training dataset; not robot safety limits.",
        "margin_fraction": margin_fraction,
        "lower": dict(zip(_ACTION_NAMES, lower.tolist(), strict=True)),
        "upper": dict(zip(_ACTION_NAMES, upper.tolist(), strict=True)),
        "any_dimension_fraction": float(valid_action_outside.sum() / denominator),
        "per_dimension_fraction": {
            name: float(valid_outside[..., i].sum() / denominator) for i, name in enumerate(_ACTION_NAMES)
        },
    }


def _load_action_range(dataset_root: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    stats_path = dataset_root / "meta" / "stats.json"
    with stats_path.open() as file:
        action_stats = json.load(file)["action"]
    action_min = np.asarray(action_stats["min"], dtype=np.float64)
    action_max = np.asarray(action_stats["max"], dtype=np.float64)
    if action_min.shape != (7,) or action_max.shape != (7,):
        raise ValueError(f"Expected 7D action stats in {stats_path}, got {action_min.shape} and {action_max.shape}")
    return action_min, action_max


def _sample_indices(dataset_size: int, num_samples: int, seed: int) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples > dataset_size:
        raise ValueError(f"num_samples ({num_samples}) exceeds dataset size ({dataset_size})")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_size, size=num_samples, replace=False))


def _checkpoint_label(checkpoint_dir: pathlib.Path) -> str:
    return f"{checkpoint_dir.parent.name}_step_{checkpoint_dir.name}"


def evaluate(args: Args) -> tuple[pathlib.Path, pathlib.Path | None]:
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_dir}")

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.root is None:
        raise ValueError(f"{args.config_name} does not define a local dataset root")
    dataset_root = pathlib.Path(data_config.root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_root}")

    dataset = data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    indices = _sample_indices(len(dataset), args.num_samples, args.seed)
    action_min, action_max = _load_action_range(dataset_root)

    logging.info("Loading checkpoint %s", args.checkpoint_dir)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.num_inference_steps},
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    wall_times_ms: list[float] = []
    rng = np.random.default_rng(args.seed)

    for sample_index in tqdm.tqdm(indices, desc="Offline inference"):
        sample = dataset[int(sample_index)]
        observation = {
            "observation/state": _to_numpy(sample["observation.state"]),
            "observation/image": _to_numpy(sample["observation.images.table"]),
            "observation/wrist_image": _to_numpy(sample["observation.images.wrist"]),
            "prompt": sample["prompt"],
        }
        noise = rng.standard_normal(
            (train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32
        )
        start = time.perf_counter()
        result = policy.infer(observation, noise=noise)
        wall_times_ms.append((time.perf_counter() - start) * 1000.0)

        predicted = np.asarray(result["actions"], dtype=np.float64)
        target = _to_numpy(sample["action"]).astype(np.float64)
        valid = ~_to_numpy(sample["action_is_pad"]).astype(bool)
        expected_shape = (train_config.model.action_horizon, 7)
        if predicted.shape != expected_shape or target.shape != expected_shape:
            raise ValueError(
                f"Expected predicted and target actions with shape {expected_shape}; "
                f"got {predicted.shape} and {target.shape} at dataset index {sample_index}"
            )
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"Non-finite prediction at dataset index {sample_index}")

        predictions.append(predicted)
        targets.append(target)
        valid_masks.append(valid)
        episode_indices.append(int(_to_numpy(sample["episode_index"])))
        frame_indices.append(int(_to_numpy(sample["frame_index"])))

    predicted_array = np.stack(predictions)
    target_array = np.stack(targets)
    valid_array = np.stack(valid_masks)
    summary = {
        "evaluation_type": "offline_training_data_action_chunk_comparison",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "config_name": args.config_name,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_size": len(dataset),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "sample_indices": indices.tolist(),
        "num_inference_steps": args.num_inference_steps,
        "action_horizon": train_config.model.action_horizon,
        "jax_devices": [str(device) for device in jax.devices()],
        "valid_action_count": int(valid_array.sum()),
        "padded_action_count_excluded": int((~valid_array).sum()),
        "metrics": _action_metrics(predicted_array, target_array, valid_array, args.gripper_threshold),
        "observed_training_range": _training_range_metrics(
            predicted_array,
            valid_array,
            action_min,
            action_max,
            args.range_margin_fraction,
        ),
        "inference_wall_time_ms": {
            "first_call_including_compilation": wall_times_ms[0],
            "steady_state_after_first_call": (
                _scalar_summary(np.asarray(wall_times_ms[1:])) if len(wall_times_ms) > 1 else None
            ),
            "all_calls": _scalar_summary(np.asarray(wall_times_ms)),
        },
        "notes": [
            "This evaluates imitation of recorded training actions, not task success or closed-loop safety.",
            "The first timing includes JAX compilation and is excluded from the steady-state timing summary.",
            "Padded actions at episode ends are excluded from metrics.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = _checkpoint_label(args.checkpoint_dir)
    summary_path = args.output_dir / f"{label}.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    predictions_path = None
    if args.save_predictions:
        predictions_path = args.output_dir / f"{label}_predictions.npz"
        np.savez_compressed(
            predictions_path,
            sample_indices=indices,
            episode_indices=np.asarray(episode_indices),
            frame_indices=np.asarray(frame_indices),
            predicted_actions=predicted_array,
            target_actions=target_array,
            valid_action_mask=valid_array,
            inference_wall_time_ms=np.asarray(wall_times_ms),
        )

    first = summary["metrics"]["first_action"]
    all_actions = summary["metrics"]["all_valid_actions"]
    print(f"Summary: {summary_path}")
    if predictions_path is not None:
        print(f"Predictions: {predictions_path}")
    print(f"First action position error: {first['position_l2_m']['mean']:.6f} m")
    print(f"First action orientation error: {first['orientation_geodesic_deg_mean']:.3f} deg")
    print(f"First action gripper accuracy: {first['gripper_binary_accuracy']:.3%}")
    print(f"All valid action position error: {all_actions['position_l2_m']['mean']:.6f} m")
    print(f"All valid action orientation error: {all_actions['orientation_geodesic_deg_mean']:.3f} deg")
    print(f"All valid action gripper accuracy: {all_actions['gripper_binary_accuracy']:.3%}")
    return summary_path, predictions_path


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    evaluate(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
