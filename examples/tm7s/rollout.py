#!/usr/bin/env python3
"""Guarded TM7S policy rollout using the proven RTRS teleoperation interfaces.

The default mode is prediction-only: S starts policy inference and simulated
15 Hz chunk consumption, X invalidates all pending work, and no TMRTC socket is
opened. Live motion additionally requires ``--enable-actuation`` and a single
Enter confirmation. ``--hold-only`` is the first live commissioning mode: S
streams only the freshly measured pose, so X/Mode-7 stop can be tested without
executing policy output.

Hardware/protocol primitives are imported from the sibling
``tm7s_keyboard_teleop`` repository so rollout and data collection use the same
TMRTS parsing, RealSense capture, workspace, guarded home, and Robotiq logic.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
import contextlib
import dataclasses
import importlib
import math
from pathlib import Path
import socket
import struct
import sys
import threading
import time
from typing import Any, NamedTuple

import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image

DEFAULT_TELEOP_ROOT = Path("/home/reinaldoyang/tm7s_keyboard_teleop")
DEFAULT_ROBOT_IP = "192.168.10.5"
DEFAULT_TABLE_CAMERA = "349222062974"
DEFAULT_WRIST_CAMERA = "349222061475"
DEFAULT_PROMPT = "Pick up the bowl and place it in the green basket"
DEFAULT_CAPTURE_DIR = Path("rollout_captures")

POLICY_RATE_HZ = 15.0
POLICY_HORIZON = 16
CONTROL_INTERVAL_MS = 67  # TMRTC requires an integer interval; approximately 15 Hz.
ACCELERATION_TIME_MS = 2000
GAIN = 1.0

# Linear speed matches the keyboard collector (0.75 mm x 100 Hz = 75 mm/s).
# Acceleration lets the TCP reach full speed within one ~1 s action chunk.
# These are PC-side command limits, not safety-rated limits.
DEFAULT_MAX_LINEAR_SPEED_MM_S = 75.0
DEFAULT_MAX_LINEAR_ACCEL_MM_S2 = 300.0
DEFAULT_MAX_ANGULAR_SPEED_DEG_S = 2.0
DEFAULT_MAX_ANGULAR_ACCEL_DEG_S2 = 10.0

# Extra z allowance beyond the measured workspace bounds. The home pose sits
# at the z ceiling, so tiny predicted lifts would otherwise stop the rollout.
Z_WORKSPACE_TOLERANCE_MM = 0.5
# Policy targets at most this far outside the workspace are clipped to its
# edge. Targets further out still trigger a software stop.
NEAR_MISS_CLIP_MM = 20.0
# Rollout-only raise of the z ceiling. The teleop ceiling already has a 10 mm
# inward margin from tune_workspace_limits, so this keeps 5 mm of that margin.
Z_CEILING_EXTENSION_MM = 5.0

MAX_FEEDBACK_AGE_SECONDS = 0.25
MAX_CAMERA_AGE_SECONDS = 0.25
MAX_CAMERA_SKEW_SECONDS = 0.10
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 10.0
INFERENCE_SETTLED_POSITION_MM = 1.0
INFERENCE_SETTLED_ORIENTATION_DEG = 0.2
TMRTC_ACK_TIMEOUT_SECONDS = 2.0

GRIPPER_OPEN_WIDTH_MM = 85.0
GRIPPER_CLOSED_WIDTH_MM = 6.5
GRIPPER_OPEN_THRESHOLD = 0.30
GRIPPER_CLOSE_THRESHOLD = 0.70
GRIPPER_COMMAND_TIMEOUT_SECONDS = 15.0
GRIPPER_SPEED = 100
GRIPPER_FORCE = 10

HOME_CONFIRMATION_SECONDS = 3.0

GUI_WIDTH = 1280
GUI_HEIGHT = 720
GUI_PREVIEW_WIDTH = 600
GUI_PREVIEW_HEIGHT = 450
GUI_PREVIEW_Y = 42
GUI_TABLE_X = 30
GUI_WRIST_X = 650
GUI_PREVIEW_RATE_HZ = 15.0


class HardwareModules(NamedTuple):
    xyz: Any
    collection: Any
    hold: Any


def load_hardware_modules(root: Path) -> HardwareModules:
    """Load the verified teleoperation implementation from its repository."""
    root = root.expanduser().resolve()
    if not (root / "teleoperation" / "tmrtc_data_collection.py").is_file():
        raise RuntimeError(f"TM7S teleoperation repository not found at {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return HardwareModules(
        importlib.import_module("teleoperation.tmrtc_cartesian_xyz_teleop"),
        importlib.import_module("teleoperation.tmrtc_data_collection"),
        importlib.import_module("testing.tmrtc_cartesian_hold_test"),
    )


def rollout_workspace_max(modules: HardwareModules) -> tuple[float, float, float]:
    x_max, y_max, z_max = modules.xyz.WORKSPACE_MAX_ROBOTBASE_MM
    return (x_max, y_max, z_max + Z_CEILING_EXTENSION_MM)


def clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm == 0.0:
        return vector
    return vector * (maximum / norm)


@dataclasses.dataclass
class VectorRateLimiter:
    """Limit vector-valued target speed and command acceleration."""

    value: np.ndarray
    max_speed: float
    max_acceleration: float
    velocity: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=np.float64).copy()
        if self.value.ndim != 1 or not np.all(np.isfinite(self.value)):
            raise ValueError("rate-limiter value must be a finite vector")
        if self.max_speed <= 0.0 or self.max_acceleration <= 0.0:
            raise ValueError("speed and acceleration limits must be positive")
        self.velocity = np.zeros_like(self.value)

    def step(self, target: Sequence[float], dt: float) -> np.ndarray:
        target_array = np.asarray(target, dtype=np.float64)
        if target_array.shape != self.value.shape or not np.all(np.isfinite(target_array)):
            raise ValueError("rate-limiter target has invalid shape or values")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("rate-limiter timestep must be positive")

        delta = target_array - self.value
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            desired_velocity = np.zeros_like(self.velocity)
        else:
            direction = delta / distance
            stopping_speed = math.sqrt(2.0 * self.max_acceleration * distance)
            desired_speed = min(self.max_speed, distance / dt, stopping_speed)
            desired_velocity = direction * desired_speed

        velocity_delta = clamp_norm(
            desired_velocity - self.velocity,
            self.max_acceleration * dt,
        )
        next_velocity = clamp_norm(self.velocity + velocity_delta, self.max_speed)
        step = next_velocity * dt
        self.value += step
        self.velocity = next_velocity
        return self.value.copy()

    def hold(self) -> None:
        self.velocity.fill(0.0)

    def decelerate(self, dt: float) -> np.ndarray:
        """Bring command velocity to zero without an ordinary-mode jump."""
        velocity_delta = clamp_norm(-self.velocity, self.max_acceleration * dt)
        self.velocity += velocity_delta
        self.value += self.velocity * dt
        return self.value.copy()

    @property
    def stationary(self) -> bool:
        return float(np.linalg.norm(self.velocity)) <= 1e-9


@dataclasses.dataclass(frozen=True)
class SafetyConfig:
    workspace_min_mm: tuple[float, float, float]
    workspace_max_mm: tuple[float, float, float]
    max_linear_speed_mm_s: float
    max_linear_accel_mm_s2: float
    max_angular_speed_deg_s: float
    max_angular_accel_deg_s2: float
    lock_orientation: bool = True

    def validate_target(self, pose_mm_deg: Sequence[float]) -> None:
        pose = np.asarray(pose_mm_deg, dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError("policy produced a non-finite or non-6D TCP target")
        tolerance = (0.0, 0.0, Z_WORKSPACE_TOLERANCE_MM)
        if any(
            pose[axis] < self.workspace_min_mm[axis] - tolerance[axis]
            or pose[axis] > self.workspace_max_mm[axis] + tolerance[axis]
            for axis in range(3)
        ):
            raise RuntimeError(
                "policy target is outside the configured RobotBase workspace: "
                + np.array2string(pose[:3], precision=3)
                + " mm"
            )

    def clip_near_miss(self, pose_mm_deg: Sequence[float]) -> tuple[np.ndarray, bool]:
        """Clip positions within NEAR_MISS_CLIP_MM outside the box to its edge."""
        pose = np.asarray(pose_mm_deg, dtype=np.float64).copy()
        low = np.asarray(self.workspace_min_mm)
        high = np.asarray(self.workspace_max_mm)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            return pose, False
        if np.any(pose[:3] < low - NEAR_MISS_CLIP_MM) or np.any(pose[:3] > high + NEAR_MISS_CLIP_MM):
            return pose, False  # Too far out; validate_target() will stop the rollout.
        clipped = np.clip(pose[:3], low, high)
        changed = bool(np.any(clipped != pose[:3]))
        pose[:3] = clipped
        return pose, changed


class CommandFilter:
    """Convert policy actions and enforce command-space rate limits."""

    def __init__(self, start_pose_mm_deg: Sequence[float], safety: SafetyConfig) -> None:
        start = np.asarray(start_pose_mm_deg, dtype=np.float64)
        if start.shape != (6,) or not np.all(np.isfinite(start)):
            raise ValueError("start pose must contain six finite values")
        self.safety = safety
        self._last_clip_print = 0.0
        self.fixed_orientation = start[3:].copy()
        self.translation = VectorRateLimiter(
            start[:3],
            safety.max_linear_speed_mm_s,
            safety.max_linear_accel_mm_s2,
        )
        self.orientation = VectorRateLimiter(
            start[3:],
            safety.max_angular_speed_deg_s,
            safety.max_angular_accel_deg_s2,
        )

    @property
    def pose(self) -> tuple[float, ...]:
        return (*self.translation.value, *self.orientation.value)

    def step(self, policy_action: Sequence[float], dt: float) -> tuple[tuple[float, ...], float]:
        action = np.asarray(policy_action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeError("policy action must contain seven finite values")
        target = np.concatenate([action[:3] * 1000.0, np.rad2deg(action[3:6])])
        if self.safety.lock_orientation:
            target[3:] = self.fixed_orientation
        raw_position = target[:3].copy()
        target, clipped = self.safety.clip_near_miss(target)
        if clipped and time.monotonic() - self._last_clip_print >= 1.0:
            self._last_clip_print = time.monotonic()
            print(
                "Clipped near-miss policy target "
                + np.array2string(raw_position, precision=3)
                + " -> "
                + np.array2string(target[:3], precision=3)
                + " mm"
            )
        self.safety.validate_target(target)
        translation = self.translation.step(target[:3], dt)
        orientation = self.orientation.step(target[3:], dt)
        command = (*translation, *orientation)
        self.safety.validate_target(command)
        return command, float(action[6])

    def hold(self) -> None:
        self.translation.hold()
        self.orientation.hold()

    def decelerate(self, dt: float) -> tuple[float, ...]:
        translation = self.translation.decelerate(dt)
        orientation = self.orientation.decelerate(dt)
        command = (*translation, *orientation)
        self.safety.validate_target(command)
        return command

    @property
    def stationary(self) -> bool:
        return self.translation.stationary and self.orientation.stationary


class FeedbackSnapshot(NamedTuple):
    state: Any
    monotonic_ns: int


class FeedbackReceiver:
    """Own TMRTS reads in a background thread and retain the newest state."""

    def __init__(
        self,
        sock: socket.socket,
        modules: HardwareModules,
        initial_state: Any,
        safety: SafetyConfig,
    ) -> None:
        self.sock = sock
        self.modules = modules
        self.safety = safety
        self._latest = FeedbackSnapshot(initial_state, time.monotonic_ns())
        self._previous_state = initial_state
        self._error: str | None = None
        self._buffer = b""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("feedback receiver already started")
        self.sock.settimeout(0.1)
        self._thread = threading.Thread(target=self._run, name="tmrts-feedback", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    chunk = self.sock.recv(4096)
                except TimeoutError:
                    continue
                if not chunk:
                    raise ConnectionError("robot closed the TMRTS connection")
                self._buffer += chunk
                packets, self._buffer = self.modules.xyz.extract_complete_packets(self._buffer)
                for packet in packets:
                    if not packet.startswith(b"$TMRTS,"):
                        continue
                    state = self.modules.hold.decode_state_packet(packet)
                    if state is None:
                        continue
                    self.modules.hold.check_robot_base(state)
                    self.safety.validate_target(state.base_pose)
                    joint_change = max(
                        abs(actual - previous)
                        for actual, previous in zip(
                            state.joints,
                            self._previous_state.joints,
                            strict=True,
                        )
                    )
                    if joint_change > self.modules.xyz.MAX_JOINT_CHANGE_PER_SAMPLE_DEG:
                        raise RuntimeError(
                            "sudden joint feedback change exceeded "
                            f"{self.modules.xyz.MAX_JOINT_CHANGE_PER_SAMPLE_DEG:.1f} deg/sample "
                            f"({joint_change:.3f} deg)"
                        )
                    self._previous_state = state
                    with self._lock:
                        self._latest = FeedbackSnapshot(state, time.monotonic_ns())
        except Exception as exc:
            with self._lock:
                self._error = f"TMRTS feedback failed: {exc}"
            self._stop.set()

    def snapshot(self, max_age_seconds: float = MAX_FEEDBACK_AGE_SECONDS) -> FeedbackSnapshot:
        with self._lock:
            latest = self._latest
            error = self._error
        if error is not None:
            raise RuntimeError(error)
        age = (time.monotonic_ns() - latest.monotonic_ns) / 1e9
        if age > max_age_seconds:
            raise RuntimeError(f"TMRTS feedback is stale ({age:.3f} s)")
        return latest

    def stop(self) -> bytes:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
            if self._thread.is_alive():
                raise RuntimeError("TMRTS feedback receiver did not stop")
        return self._buffer


class GripperSnapshot(NamedTuple):
    state: str
    width_mm: float
    busy: bool
    last_action: str
    error: str | None


class RolloutGripper:
    """Single-owner Robotiq worker supporting explicit OPEN/CLOSE requests."""

    def __init__(self, modules: HardwareModules, client: Any, initial_status: Any) -> None:
        self.modules = modules
        self.client = client
        self._status = initial_status
        self._requested: str | None = None
        self._busy = False
        self._last_action = "NONE"
        self._error: str | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="rollout-gripper", daemon=False)
        self._thread.start()

    def request(self, action: str) -> bool:
        if action not in {"OPEN", "CLOSE"}:
            raise ValueError("gripper action must be OPEN or CLOSE")
        with self._lock:
            if self._error is not None:
                return False
            if self._requested == action or (self._busy and self._last_action == action):
                return True
            self._requested = action
            self._busy = True
            self._last_action = action
        self._wake.set()
        return True

    def request_open(self) -> bool:
        return self.request("OPEN")

    def request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def join(self, timeout: float) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def snapshot(self) -> GripperSnapshot:
        with self._lock:
            return GripperSnapshot(
                self.modules.xyz.state_text(self._status),
                float(self._status.width_mm),
                self._busy,
                self._last_action,
                self._error,
            )

    def wait_until_idle(self, timeout: float, keyboard: Any | None = None) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if keyboard is not None:
                keyboard.poll()
                if keyboard.stop_requested:
                    raise RuntimeError("operator requested stop while waiting for gripper")
            snapshot = self.snapshot()
            if snapshot.error is not None:
                raise RuntimeError(snapshot.error)
            if not snapshot.busy:
                return
            time.sleep(0.02)
        raise TimeoutError("timed out waiting for gripper")

    def _move(self, action: str) -> None:
        xyz = self.modules.xyz
        status = xyz.read_status(self.client)
        if status.fault or not status.activated or status.activation_status != 3:
            raise RuntimeError("Robotiq gripper is not activated and ready")
        width = GRIPPER_OPEN_WIDTH_MM if action == "OPEN" else GRIPPER_CLOSED_WIDTH_MM
        target_raw = xyz.width_mm_to_raw_position(width)
        speed_force = (GRIPPER_SPEED << 8) | GRIPPER_FORCE
        xyz.write_registers(self.client, [0x0900, target_raw, speed_force])
        deadline = time.monotonic() + GRIPPER_COMMAND_TIMEOUT_SECONDS
        while not self._stop.is_set() and time.monotonic() < deadline:
            status = xyz.read_status(self.client)
            with self._lock:
                self._status = status
            if status.fault:
                raise RuntimeError(f"Robotiq fault 0x{status.fault:02X}")
            if status.requested_raw == target_raw and not status.moving:
                return
            self._stop.wait(0.1)
        if not self._stop.is_set():
            raise TimeoutError(f"Robotiq {action.lower()} command timed out")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    requested = self._requested
                    self._requested = None
                if requested is not None:
                    self._move(requested)
                    with self._lock:
                        self._busy = False
                    continue
                status = self.modules.xyz.read_status(self.client)
                with self._lock:
                    self._status = status
                if status.fault:
                    raise RuntimeError(f"Robotiq fault 0x{status.fault:02X}")
                self._wake.wait(0.1)
                self._wake.clear()
        except Exception as exc:
            with self._lock:
                self._busy = False
                self._error = f"Robotiq worker failed: {exc}"
            self._stop.set()


class ObservationSource:
    def __init__(
        self,
        feedback: FeedbackReceiver,
        cameras: Any,
        gripper: RolloutGripper,
        prompt: str,
    ) -> None:
        self.feedback = feedback
        self.cameras = cameras
        self.gripper = gripper
        self.prompt = prompt

    def capture(self) -> tuple[dict[str, Any], int]:
        feedback = self.feedback.snapshot()
        table, wrist = self.cameras.snapshots()
        now_ns = time.monotonic_ns()
        table_age = (now_ns - table.host_monotonic_ns) / 1e9
        wrist_age = (now_ns - wrist.host_monotonic_ns) / 1e9
        skew = abs(table.host_monotonic_ns - wrist.host_monotonic_ns) / 1e9
        if max(table_age, wrist_age) > MAX_CAMERA_AGE_SECONDS:
            raise RuntimeError(f"camera frame is stale (table={table_age:.3f}s, wrist={wrist_age:.3f}s)")
        if skew > MAX_CAMERA_SKEW_SECONDS:
            raise RuntimeError(f"camera skew is too large ({skew:.3f}s)")
        gripper = self.gripper.snapshot()
        if gripper.error is not None:
            raise RuntimeError(gripper.error)
        span = GRIPPER_OPEN_WIDTH_MM - GRIPPER_CLOSED_WIDTH_MM
        gripper_state = np.clip((GRIPPER_OPEN_WIDTH_MM - gripper.width_mm) / span, 0.0, 1.0)
        pose = feedback.state.base_pose
        state = np.asarray(
            [
                *(value / 1000.0 for value in pose[:3]),
                *(math.radians(value) for value in pose[3:]),
                gripper_state,
            ],
            dtype=np.float32,
        )
        observation = {
            "observation/state": state,
            "observation/image": np.ascontiguousarray(table.image_bgr[:, :, ::-1]),
            "observation/wrist_image": np.ascontiguousarray(wrist.image_bgr[:, :, ::-1]),
            "prompt": self.prompt,
        }
        return observation, now_ns


class InferenceResult(NamedTuple):
    generation: int
    actions: np.ndarray | None
    observation_ns: int
    elapsed_seconds: float
    error: str | None


class UnusedPolicy:
    """Placeholder that keeps hold-only commissioning server-independent."""

    def infer(self, _: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("policy inference is disabled in --hold-only mode")


class InferenceWorker:
    """Run blocking WebSocket inference without blocking keyboard stop."""

    def __init__(self, policy: Any, source: ObservationSource) -> None:
        self.policy = policy
        self.source = source
        self._request: int | None = None
        self._result: InferenceResult | None = None
        self._busy_since: float | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)
        self._thread.start()

    def request(self, generation: int) -> bool:
        with self._lock:
            if self._request is not None or self._busy_since is not None:
                return False
            self._request = generation
        self._wake.set()
        return True

    def cancel(self) -> None:
        with self._lock:
            self._request = None
            self._result = None

    def busy_seconds(self) -> float | None:
        with self._lock:
            if self._busy_since is None:
                return None
            return time.monotonic() - self._busy_since

    def take_result(self) -> InferenceResult | None:
        with self._lock:
            result = self._result
            self._result = None
            return result

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            with self._lock:
                generation = self._request
                self._request = None
                if generation is not None:
                    self._busy_since = time.monotonic()
            if generation is None:
                continue
            started = time.monotonic()
            observation_ns = time.monotonic_ns()
            try:
                observation, observation_ns = self.source.capture()
                output = self.policy.infer(observation)
                actions = np.asarray(output["actions"], dtype=np.float64)
                if actions.shape != (POLICY_HORIZON, 7) or not np.all(np.isfinite(actions)):
                    raise RuntimeError(f"policy returned invalid actions with shape {actions.shape}")
                result = InferenceResult(
                    generation,
                    actions,
                    observation_ns,
                    time.monotonic() - started,
                    None,
                )
            except Exception as exc:
                result = InferenceResult(
                    generation,
                    None,
                    observation_ns,
                    time.monotonic() - started,
                    f"policy inference failed: {exc}",
                )
            with self._lock:
                self._busy_since = None
                self._result = result


class RolloutKeyboard:
    """Focused Pygame controls compatible with guarded_joint_home()."""

    def __init__(
        self,
        pygame_module: Any,
        *,
        actuation_enabled: bool,
        table_serial: str,
        wrist_serial: str,
    ) -> None:
        self.pygame = pygame_module
        self.actuation_enabled = actuation_enabled
        self.table_serial = table_serial
        self.wrist_serial = wrist_serial
        self.screen: Any | None = None
        self.title_font: Any | None = None
        self.font: Any | None = None
        self._latest_table: Any | None = None
        self._latest_wrist: Any | None = None
        self.start_requested = False
        self.x_requested = False
        self.home_requested = False
        self.stop_requested = False
        self.focused = False
        self.pressed_keys: set[str] = set()
        self.orientation_enabled = False
        self._held: set[str] = set()
        self.home_motion_active = False
        self.home_abort_requested = False
        self.status = "STOPPED"
        self.detail = "S: start   X: software stop   H twice: home   Esc: exit"

    def __enter__(self) -> RolloutKeyboard:
        if self.pygame is None:
            raise RuntimeError("Pygame is required; install the tm7s-rollout dependency group")
        self.pygame.init()
        self.screen = self.pygame.display.set_mode((GUI_WIDTH, GUI_HEIGHT))
        self.pygame.display.set_caption("TM7S OpenPI Rollout")
        self.title_font = self.pygame.font.Font(None, 28)
        self.font = self.pygame.font.Font(None, 23)
        self.draw()
        return self

    def __exit__(self, *_: Any) -> None:
        self.pygame.quit()

    def set_status(self, status: str, detail: str) -> None:
        self.status = status
        self.detail = detail
        self.draw()

    def _camera_surface(self, frame: Any) -> Any:
        rgb = np.ascontiguousarray(frame.image_bgr[:, :, ::-1])
        height, width = rgb.shape[:2]
        surface = self.pygame.image.fromstring(rgb.tobytes(), (width, height), "RGB")
        return self.pygame.transform.smoothscale(
            surface,
            (GUI_PREVIEW_WIDTH, GUI_PREVIEW_HEIGHT),
        )

    def draw(self) -> None:
        if self.screen is None or self.font is None or self.title_font is None:
            return
        self.screen.fill((28, 31, 38))
        text_color = (235, 238, 245)
        placeholder = (53, 58, 68)
        labels = (
            (f"TABLE  {self.table_serial}", GUI_TABLE_X),
            (f"WRIST  {self.wrist_serial}", GUI_WRIST_X),
        )
        for label, x_position in labels:
            self.screen.blit(
                self.title_font.render(label, True, text_color),  # noqa: FBT003
                (x_position, 12),
            )

        for frame, x_position in (
            (self._latest_table, GUI_TABLE_X),
            (self._latest_wrist, GUI_WRIST_X),
        ):
            target = (
                x_position,
                GUI_PREVIEW_Y,
                GUI_PREVIEW_WIDTH,
                GUI_PREVIEW_HEIGHT,
            )
            if frame is None:
                self.pygame.draw.rect(self.screen, placeholder, target)
                waiting = self.title_font.render(
                    "Waiting for camera...",
                    True,  # noqa: FBT003
                    text_color,
                )
                waiting_rect = waiting.get_rect(
                    center=(
                        x_position + GUI_PREVIEW_WIDTH // 2,
                        GUI_PREVIEW_Y + GUI_PREVIEW_HEIGHT // 2,
                    )
                )
                self.screen.blit(waiting, waiting_rect)
            else:
                self.screen.blit(
                    self._camera_surface(frame),
                    (x_position, GUI_PREVIEW_Y),
                )

        mode = "LIVE ACTUATION" if self.actuation_enabled else "PREDICTION ONLY"
        lines = (
            f"TM7S POLICY ROLLOUT — {mode} — State: {self.status}",
            self.detail,
            "S: start   X: software stop   H twice: guarded home   Escape: stop and exit",
            "Keep this window focused. X is not a safety-rated emergency stop.",
        )
        colors = (
            (255, 190, 80),
            (220, 225, 232),
            (235, 238, 245),
            (255, 100, 100),
        )
        for index, (line, color) in enumerate(zip(lines, colors, strict=True)):
            self.screen.blit(
                self.font.render(line, True, color),  # noqa: FBT003
                (30, 510 + index * 43),
            )
        self.pygame.display.flip()

    def draw_preview(
        self,
        table: Any | None,
        wrist: Any | None,
        **__: Any,
    ) -> None:
        self._latest_table = table
        self._latest_wrist = wrist
        self.draw()

    def poll(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.stop_requested = True
            elif event.type == self.pygame.KEYDOWN:
                key = self.pygame.key.name(event.key).lower()
                if key in self._held:
                    continue
                self._held.add(key)
                if key == "escape":
                    self.stop_requested = True
                elif key == "s":
                    self.start_requested = True
                elif key == "x":
                    self.x_requested = True
                    if self.home_motion_active:
                        # guarded_joint_home watches stop_requested. During a
                        # home trajectory, temporarily use that signal to make
                        # it stop TMRTC. home() clears it after recovering the
                        # stopped feedback/UI session.
                        self.home_abort_requested = True
                        self.stop_requested = True
                elif key == "h":
                    self.home_requested = True
            elif event.type == self.pygame.KEYUP:
                self._held.discard(self.pygame.key.name(event.key).lower())
            elif event.type == self.pygame.WINDOWFOCUSLOST:
                self._held.clear()
        self.focused = bool(self.pygame.key.get_focused())

    def consume(self, name: str) -> bool:
        value = bool(getattr(self, name))
        setattr(self, name, False)
        return value


class RolloutRuntime:
    def __init__(
        self,
        modules: HardwareModules,
        keyboard: RolloutKeyboard,
        cameras: Any,
        gripper: RolloutGripper,
        tmrts_sock: socket.socket,
        feedback: FeedbackReceiver,
        tmrtc_sock: socket.socket | None,
        inference: InferenceWorker,
        safety: SafetyConfig,
        *,
        actuation_enabled: bool,
        hold_only: bool,
        inference_timeout_seconds: float,
        capture_dir: Path,
    ) -> None:
        self.capture_dir = capture_dir
        self.modules = modules
        self.keyboard = keyboard
        self.cameras = cameras
        self.gripper = gripper
        self.tmrts_sock = tmrts_sock
        self.feedback = feedback
        self.tmrtc_sock = tmrtc_sock
        self.inference = inference
        self.safety = safety
        self.actuation_enabled = actuation_enabled
        self.hold_only = hold_only
        self.inference_timeout_seconds = inference_timeout_seconds
        self.generation = 0
        self.running = False
        self.mode_active = False
        self.rtc_buffer = b""
        self.filter: CommandFilter | None = None
        self.actions: deque[np.ndarray] = deque()
        self.next_tick = time.monotonic()
        self.inference_requested = False
        self.gripper_closed = gripper.snapshot().width_mm <= (GRIPPER_OPEN_WIDTH_MM + GRIPPER_CLOSED_WIDTH_MM) / 2.0
        self.home_armed_until = 0.0

    def _build_cartesian_setting(self, transaction_id: bytes) -> bytes:
        settings = b"SC" + struct.pack("<ifi", ACCELERATION_TIME_MS, GAIN, CONTROL_INTERVAL_MS)
        return self.modules.xyz.build_tmrtc_packet(transaction_id + b",8," + settings)

    def _capture_stable(self) -> Any:
        self.feedback.stop()
        state = self.modules.hold.capture_stable_state(self.tmrts_sock)
        self.feedback = FeedbackReceiver(self.tmrts_sock, self.modules, state, self.safety)
        self.feedback.start()
        self.inference.source.feedback = self.feedback
        return state

    def _start_live_hold(self) -> None:
        if not self.actuation_enabled or self.tmrtc_sock is None:
            return
        state = self._capture_stable()
        self.safety.validate_target(state.base_pose)
        set_id = b"SETPOL"
        start_id = b"STARTPOL"
        self.tmrtc_sock.settimeout(3.0)
        self.tmrtc_sock.sendall(self._build_cartesian_setting(set_id))
        self.rtc_buffer = self.modules.xyz.wait_for_tmrtc_ack(self.tmrtc_sock, set_id, self.rtc_buffer)
        self.tmrtc_sock.sendall(self.modules.xyz.build_tmrtc_start(start_id))
        self.rtc_buffer = self.modules.xyz.wait_for_tmrtc_ack(self.tmrtc_sock, start_id, self.rtc_buffer)
        self.tmrtc_sock.settimeout(0.25)
        self.mode_active = True
        self.filter = CommandFilter(state.base_pose, self.safety)
        self.next_tick = time.monotonic()

    def _save_start_frames(self) -> None:
        """Save the table and wrist frames seen when S is pressed."""
        try:
            table, wrist = self.cameras.snapshots()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in (("table", table), ("wrist", wrist)):
                path = self.capture_dir / f"{stamp}_{name}.png"
                Image.fromarray(np.ascontiguousarray(frame.image_bgr[:, :, ::-1])).save(path)
            print(f"Saved start frames to {self.capture_dir.resolve()}/{stamp}_*.png")
        except Exception as exc:
            # A failed snapshot save must never block or fault the rollout.
            print(f"WARNING: could not save start frames: {exc}", file=sys.stderr)

    def start(self) -> None:
        if self.running:
            return
        self._save_start_frames()
        self.generation += 1
        self.actions.clear()
        self.inference.cancel()
        if self.actuation_enabled:
            self._start_live_hold()
        self.running = True
        self.inference_requested = False
        detail = "holding measured pose; X tests Mode-7 stop" if self.hold_only else "waiting for fresh inference"
        self.keyboard.set_status("RUNNING", detail)

    def software_stop(self, reason: str) -> None:
        self.generation += 1
        self.running = False
        self.actions.clear()
        self.inference.cancel()
        self.inference_requested = False
        if self.filter is not None:
            self.filter.hold()
        if self.mode_active and self.tmrtc_sock is not None:
            stop_id = b"STOPPOL"
            try:
                self.tmrtc_sock.settimeout(TMRTC_ACK_TIMEOUT_SECONDS)
                self.tmrtc_sock.sendall(self.modules.xyz.build_tmrtc_stop(stop_id))
                self.rtc_buffer = self.modules.xyz.wait_for_tmrtc_ack(
                    self.tmrtc_sock,
                    stop_id,
                    self.rtc_buffer,
                    timeout=TMRTC_ACK_TIMEOUT_SECONDS,
                )
            finally:
                self.mode_active = False
                self.filter = None
        self.keyboard.set_status("STOPPED", reason + "; press S for a fresh start")
        print(f"SOFTWARE STOP: {reason}")

    def _request_inference_if_needed(self) -> None:
        if self.hold_only or not self.running or self.actions or self.inference_requested:
            return
        if self.actuation_enabled and self.filter is not None:
            if not self.filter.stationary or self.gripper.snapshot().busy:
                return
            actual = self.feedback.snapshot().state
            position_error, orientation_error = self.modules.hold.pose_error(actual.base_pose, self.filter.pose)
            if position_error > INFERENCE_SETTLED_POSITION_MM or orientation_error > INFERENCE_SETTLED_ORIENTATION_DEG:
                return
        if self.inference.request(self.generation):
            self.inference_requested = True
            self.keyboard.set_status("INFER", "holding pose while policy predicts")

    def _process_inference(self) -> None:
        busy = self.inference.busy_seconds()
        if self.running and self.inference_requested and busy is not None and busy > self.inference_timeout_seconds:
            self.software_stop(f"inference exceeded {self.inference_timeout_seconds:.1f}s timeout")
            return
        result = self.inference.take_result()
        if result is None:
            return
        self.inference_requested = False
        if result.generation != self.generation or not self.running:
            print("Discarded inference result from an invalidated generation.")
            return
        if result.error is not None or result.actions is None:
            self.software_stop(result.error or "inference returned no actions")
            return
        age = (time.monotonic_ns() - result.observation_ns) / 1e9
        if age > self.inference_timeout_seconds:
            self.software_stop(f"inference result is stale ({age:.3f}s)")
            return
        self.actions.extend(result.actions)
        self.keyboard.set_status(
            "EXECUTE" if self.actuation_enabled else "SIMULATE",
            f"16-action chunk; inference {result.elapsed_seconds:.3f}s",
        )

    def _apply_gripper(self, value: float) -> None:
        if value >= GRIPPER_CLOSE_THRESHOLD and not self.gripper_closed:
            if not self.gripper.request("CLOSE"):
                raise RuntimeError("gripper rejected CLOSE request")
            self.gripper_closed = True
        elif value <= GRIPPER_OPEN_THRESHOLD and self.gripper_closed:
            if not self.gripper.request("OPEN"):
                raise RuntimeError("gripper rejected OPEN request")
            self.gripper_closed = False

    def _tick(self) -> None:
        if not self.running:
            return
        if self.actuation_enabled and self.mode_active:
            if self.filter is None or self.tmrtc_sock is None:
                raise RuntimeError("live mode has no command filter or TMRTC socket")
            if self.actions:
                pose, gripper_value = self.filter.step(self.actions.popleft(), 1.0 / POLICY_RATE_HZ)
                self._apply_gripper(gripper_value)
            else:
                pose = self.filter.decelerate(1.0 / POLICY_RATE_HZ)
            self.tmrtc_sock.sendall(self.modules.xyz.build_tmrtc_pose_target(pose))
        elif self.actions:
            self.actions.popleft()

        if not self.actions and not self.hold_only:
            self._request_inference_if_needed()

    def home(self) -> None:
        now = time.monotonic()
        if not self.actuation_enabled or self.tmrtc_sock is None:
            self.home_armed_until = 0.0
            print("HOME ignored in prediction-only mode.")
            return
        if now > self.home_armed_until:
            self.home_armed_until = now + HOME_CONFIRMATION_SECONDS
            self.keyboard.set_status(
                "HOME ARMED",
                f"press H again within {HOME_CONFIRMATION_SECONDS:.0f}s",
            )
            return

        self.home_armed_until = 0.0
        self.keyboard.home_motion_active = True
        session = None
        feedback_stopped = False
        try:
            self.software_stop("home requested")
            self.gripper.request_open()
            self.keyboard.set_status("HOME", "opening gripper before guarded joint home")
            self.gripper.wait_until_idle(GRIPPER_COMMAND_TIMEOUT_SECONDS, self.keyboard)
            self.gripper_closed = False

            self.feedback.stop()
            feedback_stopped = True
            session = self.modules.collection.MotionSession(
                stream=None,
                rtc_buffer=self.rtc_buffer,
                mode_active=False,
                mode_name="none",
            )
            current_state = self.modules.hold.capture_stable_state(self.tmrts_sock)
            settled, _ = self.modules.collection.guarded_joint_home(
                self.tmrts_sock,
                self.tmrtc_sock,
                self.keyboard,
                session,
                current_state,
                b"",
                self.gripper,
                self.cameras,
                self.modules.xyz.MAX_ORIENTATION_FROM_START_DEG,
            )
            self.modules.collection.stop_active_mode(self.tmrtc_sock, session, b"STOPHOLD")
            self.rtc_buffer = session.rtc_buffer
            self.feedback = FeedbackReceiver(self.tmrts_sock, self.modules, settled, self.safety)
            self.feedback.start()
            self.inference.source.feedback = self.feedback
            self.keyboard.set_status("STOPPED", "home reached; gripper open; press S to start")
        except Exception:
            if not self.keyboard.home_abort_requested:
                raise
            if session is not None:
                if session.mode_active:
                    self.modules.collection.stop_active_mode(
                        self.tmrtc_sock,
                        session,
                        b"STOPHOMEX",
                    )
                self.rtc_buffer = session.rtc_buffer
            if feedback_stopped:
                settled = self.modules.hold.capture_stable_state(self.tmrts_sock)
                self.feedback = FeedbackReceiver(
                    self.tmrts_sock,
                    self.modules,
                    settled,
                    self.safety,
                )
                self.feedback.start()
                self.inference.source.feedback = self.feedback
            self.keyboard.x_requested = False
            self.keyboard.stop_requested = False
            self.keyboard.home_abort_requested = False
            self.keyboard.set_status(
                "STOPPED",
                "home aborted by X; press S for a fresh start",
            )
            print("SOFTWARE STOP: X aborted guarded home; rollout remains open")
        finally:
            self.keyboard.home_motion_active = False

    def run(self) -> None:
        period = 1.0 / POLICY_RATE_HZ
        preview_period = 1.0 / GUI_PREVIEW_RATE_HZ
        next_preview = time.monotonic()
        while not self.keyboard.stop_requested:
            self.keyboard.poll()
            if self.keyboard.consume("x_requested"):
                self.software_stop("X pressed")
            if self.keyboard.consume("home_requested"):
                self.home()
            if self.keyboard.consume("start_requested"):
                self.start()

            try:
                self.feedback.snapshot()
                gripper = self.gripper.snapshot()
                if gripper.error is not None:
                    raise RuntimeError(gripper.error)
                self.cameras.raise_if_failed()
                now = time.monotonic()
                if now >= next_preview:
                    table, wrist = self.cameras.snapshots()
                    self.keyboard.draw_preview(table, wrist)
                    next_preview += preview_period
                    if next_preview <= now:
                        next_preview = now + preview_period
                self._process_inference()
                if self.mode_active and self.filter is not None:
                    actual = self.feedback.snapshot().state
                    position_error, orientation_error = self.modules.hold.pose_error(actual.base_pose, self.filter.pose)
                    if position_error >= self.modules.xyz.TRACKING_ABORT_MM:
                        raise RuntimeError(
                            "Cartesian tracking error exceeded "
                            f"{self.modules.xyz.TRACKING_ABORT_MM:.1f} mm "
                            f"({position_error:.3f} mm)"
                        )
                    if orientation_error > self.modules.xyz.MAX_TRACKING_ORIENTATION_ERROR_DEG:
                        raise RuntimeError(
                            "orientation tracking error exceeded "
                            f"{self.modules.xyz.MAX_TRACKING_ORIENTATION_ERROR_DEG:.1f} deg "
                            f"({orientation_error:.3f} deg)"
                        )
                now = time.monotonic()
                if now >= self.next_tick:
                    self._tick()
                    self.next_tick += period
                    if self.next_tick <= now:
                        self.next_tick = now + period
            except Exception as exc:
                try:
                    self.software_stop(str(exc))
                except Exception as stop_exc:
                    self.keyboard.set_status("FAULT", f"{exc}; stop failed: {stop_exc}")
                    raise RuntimeError(f"rollout fault: {exc}; software stop failed: {stop_exc}") from exc
            time.sleep(0.005)


def initialize_gripper(modules: HardwareModules, *, allow_activation: bool) -> tuple[Any, RolloutGripper]:
    xyz = modules.xyz
    client = xyz.create_client(xyz.GRIPPER_SERIAL_PORT)
    if not client.connect():
        raise ConnectionError("Robotiq connection failed")
    status = xyz.read_status(client)
    if status.fault and status.fault != 0x09:
        raise RuntimeError(f"Robotiq fault 0x{status.fault:02X}")
    if status.fault == 0x09 and status.activated:
        status = xyz.wait_for_gripper_watchdog_clear(client, status)
    if not status.activated or status.activation_status == 0:
        if not allow_activation:
            raise RuntimeError(
                "Robotiq is not activated; prediction-only mode will not run the movement-producing activation cycle"
            )
        input("The gripper may move during activation. Clear its workspace, then press Enter: ")
        if not xyz.reset_and_activate(client):
            raise RuntimeError("Robotiq activation failed")
        status = xyz.read_status(client)
    if status.fault or not status.activated or status.activation_status != 3:
        raise RuntimeError("Robotiq gripper is not ready")
    controller = RolloutGripper(modules, client, status)
    controller.start()
    return client, controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--teleop-root", type=Path, default=DEFAULT_TELEOP_ROOT)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--table-camera", default=DEFAULT_TABLE_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="folder for the table/wrist frames saved each time S is pressed",
    )
    parser.add_argument("--enable-actuation", action="store_true")
    parser.add_argument(
        "--hold-only",
        action="store_true",
        help="commission X/Mode-7 stop while streaming only the measured hold pose",
    )
    parser.add_argument("--enable-policy-orientation", action="store_true")
    parser.add_argument("--max-linear-speed-mm-s", type=float, default=DEFAULT_MAX_LINEAR_SPEED_MM_S)
    parser.add_argument("--max-linear-accel-mm-s2", type=float, default=DEFAULT_MAX_LINEAR_ACCEL_MM_S2)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=DEFAULT_MAX_ANGULAR_SPEED_DEG_S)
    parser.add_argument("--max-angular-accel-deg-s2", type=float, default=DEFAULT_MAX_ANGULAR_ACCEL_DEG_S2)
    parser.add_argument("--inference-timeout-seconds", type=float, default=DEFAULT_INFERENCE_TIMEOUT_SECONDS)
    return parser


def run_self_test(modules: HardwareModules) -> None:
    safety = SafetyConfig(
        tuple(modules.xyz.WORKSPACE_MIN_ROBOTBASE_MM),
        rollout_workspace_max(modules),
        DEFAULT_MAX_LINEAR_SPEED_MM_S,
        DEFAULT_MAX_LINEAR_ACCEL_MM_S2,
        DEFAULT_MAX_ANGULAR_SPEED_DEG_S,
        DEFAULT_MAX_ANGULAR_ACCEL_DEG_S2,
    )
    start = np.asarray([0.0, 450.0, 350.0, 179.0, 0.0, 0.0])
    limiter = CommandFilter(start, safety)
    target = np.asarray([0.1, 0.45, 0.35, 0.0, 0.0, 0.0, 0.0])
    previous = np.asarray(limiter.pose[:3])
    previous_velocity = np.zeros(3)
    dt = 1.0 / POLICY_RATE_HZ
    for _ in range(100):
        pose, _ = limiter.step(target, dt)
        current = np.asarray(pose[:3])
        velocity = (current - previous) / dt
        assert np.linalg.norm(velocity) <= DEFAULT_MAX_LINEAR_SPEED_MM_S + 1e-6
        assert np.linalg.norm(velocity - previous_velocity) <= DEFAULT_MAX_LINEAR_ACCEL_MM_S2 * dt + 1e-6
        assert np.allclose(pose[3:], start[3:])
        previous = current
        previous_velocity = velocity
    reverse_target = np.asarray([-0.1, 0.45, 0.35, 0.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        pose, _ = limiter.step(reverse_target, dt)
        current = np.asarray(pose[:3])
        velocity = (current - previous) / dt
        assert np.linalg.norm(velocity) <= DEFAULT_MAX_LINEAR_SPEED_MM_S + 1e-6
        assert np.linalg.norm(velocity - previous_velocity) <= DEFAULT_MAX_LINEAR_ACCEL_MM_S2 * dt + 1e-6
        previous = current
        previous_velocity = velocity
    for _ in range(20):
        pose = limiter.decelerate(dt)
        current = np.asarray(pose[:3])
        velocity = (current - previous) / dt
        assert np.linalg.norm(velocity) <= DEFAULT_MAX_LINEAR_SPEED_MM_S + 1e-6
        assert np.linalg.norm(velocity - previous_velocity) <= DEFAULT_MAX_LINEAR_ACCEL_MM_S2 * dt + 1e-6
        previous = current
        previous_velocity = velocity
        if limiter.stationary:
            break
    assert limiter.stationary
    try:
        safety.validate_target((-1000.0, 450.0, 350.0, 179.0, 0.0, 0.0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("workspace guard did not reject an outside target")
    ceiling = safety.workspace_max_mm[2]
    near_miss, changed = safety.clip_near_miss((0.0, 450.0, ceiling + 1.1, 179.0, 0.0, 0.0))
    assert changed
    assert near_miss[2] == ceiling
    far_miss, changed = safety.clip_near_miss((0.0, 450.0, ceiling + NEAR_MISS_CLIP_MM + 1.0, 179.0, 0.0, 0.0))
    assert not changed
    try:
        safety.validate_target(far_miss)
    except RuntimeError:
        pass
    else:
        raise AssertionError("workspace guard did not reject a far-outside target")
    print("TM7S rollout self-test passed.")


def main() -> int:
    args = build_parser().parse_args()
    modules = load_hardware_modules(args.teleop_root)
    if args.self_test:
        run_self_test(modules)
        return 0
    if args.hold_only and not args.enable_actuation:
        print("--hold-only requires --enable-actuation", file=sys.stderr)
        return 2
    numeric_limits = (
        args.max_linear_speed_mm_s,
        args.max_linear_accel_mm_s2,
        args.max_angular_speed_deg_s,
        args.max_angular_accel_deg_s2,
        args.inference_timeout_seconds,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in numeric_limits):
        print("all speed, acceleration, and timeout values must be positive", file=sys.stderr)
        return 2

    safety = SafetyConfig(
        tuple(modules.xyz.WORKSPACE_MIN_ROBOTBASE_MM),
        rollout_workspace_max(modules),
        args.max_linear_speed_mm_s,
        args.max_linear_accel_mm_s2,
        args.max_angular_speed_deg_s,
        args.max_angular_accel_deg_s2,
        lock_orientation=not args.enable_policy_orientation,
    )

    policy = (
        UnusedPolicy()
        if args.hold_only
        else websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
        )
    )
    keyboard = RolloutKeyboard(
        modules.xyz.pygame,
        actuation_enabled=args.enable_actuation,
        table_serial=args.table_camera,
        wrist_serial=args.wrist_camera,
    )
    cameras = modules.collection.DualCameraRig(args.table_camera, args.wrist_camera)
    gripper_client = None
    gripper: RolloutGripper | None = None
    tmrts_sock: socket.socket | None = None
    tmrtc_sock: socket.socket | None = None
    feedback: FeedbackReceiver | None = None
    inference: InferenceWorker | None = None
    runtime: RolloutRuntime | None = None

    try:
        keyboard.__enter__()
        cameras.start()
        gripper_client, gripper = initialize_gripper(
            modules,
            allow_activation=args.enable_actuation,
        )

        if args.enable_actuation:
            print("Live mode uses a software stop only; keep the physical E-stop accessible.")
            print("Select RobotBase, verify the TCP/workspace, and press Play on the robot.")
            input("Press Enter to arm live TMRTC access (Ctrl+C cancels): ")

        tmrts_sock = socket.create_connection((args.robot_ip, modules.xyz.TMRTS_PORT), timeout=5.0)
        tmrts_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        modules.xyz.send_tmrts_setup(tmrts_sock)
        start_state = modules.hold.capture_stable_state(tmrts_sock, discard_queued=False)
        modules.hold.check_robot_base(start_state)
        safety.validate_target(start_state.base_pose)
        feedback = FeedbackReceiver(tmrts_sock, modules, start_state, safety)
        feedback.start()

        if args.enable_actuation:
            tmrtc_sock = socket.create_connection((args.robot_ip, modules.xyz.TMRTC_PORT), timeout=5.0)
            tmrtc_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            tmrtc_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        source = ObservationSource(feedback, cameras, gripper, args.prompt)
        inference = InferenceWorker(policy, source)
        runtime = RolloutRuntime(
            modules,
            keyboard,
            cameras,
            gripper,
            tmrts_sock,
            feedback,
            tmrtc_sock,
            inference,
            safety,
            actuation_enabled=args.enable_actuation,
            hold_only=args.hold_only,
            inference_timeout_seconds=args.inference_timeout_seconds,
            capture_dir=args.capture_dir,
        )
        runtime.run()
        runtime.software_stop("rollout window closed")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"TM7S rollout failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None and runtime.mode_active:
            try:
                runtime.software_stop("rollout shutting down")
            except Exception as exc:
                print(f"WARNING: could not confirm TMRTC stop: {exc}", file=sys.stderr)
        if inference is not None:
            inference.stop()
        if tmrtc_sock is not None:
            tmrtc_sock.close()
        active_feedback = runtime.feedback if runtime is not None else feedback
        if active_feedback is not None:
            with contextlib.suppress(Exception):
                active_feedback.stop()
        if tmrts_sock is not None:
            try:
                tmrts_sock.settimeout(1.0)
                tmrts_sock.sendall(modules.xyz.build_tmrts_packet(b"2,7,0"))
            except OSError:
                pass
            tmrts_sock.close()
        cameras.stop()
        if gripper is not None:
            gripper.request_stop()
            gripper.join(1.5)
        if gripper_client is not None:
            gripper_client.close()
        keyboard.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
