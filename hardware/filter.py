#!/usr/bin/env python3
"""OCR Safety Filter node.

This node subscribes to LCM channels for sensor data and nominal commands,
applies the OCR safety filter, and publishes filtered commands.

LCM Channels:
    Subscribes:
        - LIDAR_SCAN: Raw LiDAR data
        - ROBOT_STATE: Pose estimates (x, y, theta)
        - NOMINAL_COMMAND: Velocity commands from planner

    Publishes:
        - SAFE_COMMAND: Filtered velocity commands
        - SAFETY_STATUS: Filter status for logging/visualization

Usage:
    python -m hardware.filter --config hardware/configs/default.yaml
"""

import argparse
import os
import pickle
import sys
import threading
import time

import numpy as np
import torch
import yaml
from scipy import interpolate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lcm
except ImportError:
    raise ImportError(
        "LCM not installed. Install with: pip install lcm\n"
        "See https://lcm-proj.github.io/ for more information."
    )

from hardware.lcm_types import lidar_scan, robot_state, velocity_command, safety_status

from utils.dynamics import Dubins3D
from utils.disturbance_estimator import DisturbanceEstimator
from utils.safety_filter import SafetyFilter
from utils.value_network.models import LiDARValueNN


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config: dict) -> tuple:
    """Load the OCR value network model.

    Returns:
        Tuple of (model, options, device) where device is the torch device used.
    """
    with open(config["model"]["options"], "rb") as f:
        options = pickle.load(f)

    # Determine device (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure normalization tensors are on the target device
    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    model = LiDARValueNN(
        to_device(options["input_means"]),
        to_device(options["input_stds"]),
        to_device(options["output_mean"]),
        to_device(options["output_std"]),
        input_dim=5 + options["num_rays"],
    )
    model.load_state_dict(torch.load(config["model"]["checkpoint"], map_location=device))
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, options, device


class SafetyFilterNode:
    """LCM-based safety filter node."""

    def __init__(self, config: dict, keyboard=None):
        self.config = config
        self.keyboard = keyboard

        # Load model
        print("Loading model...")
        model, options, device = load_model(config)
        self.options = options
        print(f"Using device: {device}")

        # Setup dynamics
        dynamics = Dubins3D()

        # Setup control bounds
        control_min = np.array([config["control"]["v_x_min"], config["control"]["v_yaw_min"]])
        control_max = np.array([config["control"]["v_x_max"], config["control"]["v_yaw_max"]])

        # Setup safety filter
        self.safety_filter = SafetyFilter(
            model=model,
            dynamics=dynamics,
            control_min=control_min,
            control_max=control_max,
            filter_threshold=config["filter"]["threshold"],
            calibration_value_adjustment=config["filter"]["calibration_adjustment"],
            device=device,
            slack_coeff=config["filter"]["slack_coeff"],
        )

        # Setup disturbance estimator
        dt = config["robot"]["dt"]
        prediction_steps = int(config["disturbance"]["prediction_horizon"] / dt)
        window_steps = int(config["disturbance"]["window_horizon"] / dt)
        self.disturbance_estimator = DisturbanceEstimator(
            np.array([0.0, 0.0, 0.0]),
            prediction_steps,
            window_steps,
            config["disturbance"]["coverage"],
            config["disturbance"]["std_width"],
            dynamics,
        )
        self.max_dst = np.array([config["disturbance"]["max_dxdy"], config["disturbance"]["max_dth"]])

        # LiDAR params
        self.thetas = np.linspace(-np.pi, np.pi, options["num_rays"], endpoint=False)
        self.quality_threshold = config["lidar"]["quality_threshold"]
        self.lidar_min = config["lidar"]["min_range"]
        self.lidar_max = config["lidar"]["max_range"]
        self.focus_radius = config["lidar"]["focus_radius"]
        self.focus_fov = config["lidar"]["focus_fov"]
        self.interp_radius = config["slam"]["interp_radius"]

        # Thread-safe data storage
        self.data_lock = threading.Lock()
        self.latest_lidar = None
        self.latest_lidar_state = None
        self.latest_state = None
        self.prev_state = None
        self.prev_control = None
        self.prev_time = None

        # LCM setup
        self.lc = lcm.LCM()
        self.dt = config["robot"]["dt"]

        print("Safety filter node initialized.")

    def preprocess_scan(self, scan):
        """Preprocess raw LiDAR scan for network input."""
        count = scan.count
        if count == 0:
            return None

        angles = np.array(scan.angle[:count])
        ranges = np.array(scan.range[:count])
        quality = np.array(scan.quality[:count])

        valid_mask = quality > self.quality_threshold
        if not np.any(valid_mask):
            return None

        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        sorted_indices = np.argsort(valid_angles)
        sorted_angles = valid_angles[sorted_indices]
        sorted_ranges = valid_ranges[sorted_indices]

        interp_func = interpolate.interp1d(
            sorted_angles, sorted_ranges, kind="nearest",
            bounds_error=False, fill_value=(sorted_ranges[0], sorted_ranges[-1]),
        )
        interp_ranges = interp_func(self.thetas)

        theta_dists = np.abs((self.thetas[:, np.newaxis] - sorted_angles + np.pi) % (2 * np.pi) - np.pi)
        interp_ranges[theta_dists.min(axis=-1) > self.interp_radius] = float("inf")

        inp_lidar = np.clip(interp_ranges, self.lidar_min, self.lidar_max)
        inp_lidar[np.abs(self.thetas) > self.focus_fov] = self.lidar_max
        inp_lidar_min = np.min(inp_lidar)
        inp_lidar[inp_lidar > inp_lidar_min + self.focus_radius] = self.lidar_max

        return inp_lidar

    def publish_zero_velocity(self):
        """Publish zero velocity command."""
        cmd = velocity_command()
        cmd.timestamp = int(time.time() * 1e6)
        cmd.v_x = 0.0
        cmd.v_yaw = 0.0
        cmd.is_safe = True
        self.lc.publish("SAFE_COMMAND", cmd.encode())

    def handle_lidar(self, channel, data):
        """Handle incoming LiDAR scan."""
        scan = lidar_scan.decode(data)
        with self.data_lock:
            self.latest_lidar = scan
            if self.latest_state is not None:
                self.latest_lidar_state = np.array([
                    self.latest_state.x, self.latest_state.y, self.latest_state.theta
                ])

    def handle_state(self, channel, data):
        """Handle incoming state estimate."""
        state = robot_state.decode(data)
        with self.data_lock:
            self.latest_state = state

    def handle_command(self, channel, data):
        """Handle incoming nominal command and apply safety filter."""
        cmd = velocity_command.decode(data)
        current_time = time.time()

        # Check keyboard controls
        is_paused = self.keyboard is not None and self.keyboard.is_paused
        is_immobilized = self.keyboard is not None and self.keyboard.is_immobilized

        if is_paused:
            self.publish_zero_velocity()
            with self.data_lock:
                self.prev_time = None
            return

        # Get latest data
        with self.data_lock:
            scan = self.latest_lidar
            lidar_state = self.latest_lidar_state
            state = self.latest_state
            p_state = self.prev_state
            p_control = self.prev_control
            p_time = self.prev_time

        if scan is None or state is None or lidar_state is None:
            self.publish_zero_velocity()
            return

        current_state = np.array([state.x, state.y, state.theta])

        # Update disturbance estimator
        if p_state is not None and p_control is not None and p_time is not None:
            dt_actual = current_time - p_time
            if dt_actual > 0:
                self.disturbance_estimator.store_observation(p_control, dt_actual, current_state)

        # Compute disturbance bounds
        if len(self.disturbance_estimator.controls) >= self.disturbance_estimator.prediction_steps:
            disturbance_bounds = self.disturbance_estimator.estimate_disturbance_bounds()
            disturbance_norm = np.max(np.abs(disturbance_bounds), axis=0)
            dst = np.array([np.linalg.norm(disturbance_norm[:2]), disturbance_norm[2]])
            dst = np.minimum(dst, self.max_dst)
        else:
            dst = self.max_dst / 2

        # Preprocess LiDAR
        inp_lidar = self.preprocess_scan(scan)
        if inp_lidar is None:
            self.publish_zero_velocity()
            return

        # Run safety filter
        pred_value, filtered_control = self.safety_filter.filter(
            current_state, inp_lidar, dst, cmd.v_x, cmd.v_yaw, lidar_state
        )
        v_x, v_yaw = filtered_control
        is_intervening = pred_value < self.config["filter"]["threshold"]

        # Override to zero if immobilized (but keep infrastructure running)
        if is_immobilized:
            v_x, v_yaw = 0.0, 0.0

        # Update stored state/control
        with self.data_lock:
            self.prev_state = current_state
            self.prev_control = np.array([v_x, v_yaw])
            self.prev_time = current_time

        # Publish safe command
        if is_immobilized:
            self.publish_zero_velocity()
        else:
            safe_cmd = velocity_command()
            safe_cmd.timestamp = int(current_time * 1e6)
            safe_cmd.v_x = float(v_x)
            safe_cmd.v_yaw = float(v_yaw)
            safe_cmd.is_safe = not is_intervening
            self.lc.publish("SAFE_COMMAND", safe_cmd.encode())

        # Publish status
        status = safety_status()
        status.timestamp = int(current_time * 1e6)
        status.nom_v_x = cmd.v_x
        status.nom_v_yaw = cmd.v_yaw
        status.safe_v_x = v_x
        status.safe_v_yaw = v_yaw
        status.state = list(current_state)
        status.lidar_state = list(lidar_state)
        status.value = float(pred_value)
        status.threshold = self.config["filter"]["threshold"]
        status.is_intervening = is_intervening
        status.dst_dxdy = float(dst[0])
        status.dst_dth = float(dst[1])
        status.num_rays = len(inp_lidar)
        padded_lidar = np.zeros(100)
        padded_lidar[:min(len(inp_lidar), 100)] = inp_lidar[:min(len(inp_lidar), 100)]
        status.inp_lidar = list(padded_lidar)
        self.lc.publish("SAFETY_STATUS", status.encode())

    def run(self):
        """Main loop."""
        print("Safety filter started.")
        print("Subscribed to: LIDAR_SCAN, ROBOT_STATE, NOMINAL_COMMAND")
        print("Publishing to: SAFE_COMMAND, SAFETY_STATUS")

        sub = self.lc.subscribe("LIDAR_SCAN", self.handle_lidar)
        sub.set_queue_capacity(1)
        sub = self.lc.subscribe("ROBOT_STATE", self.handle_state)
        sub.set_queue_capacity(1)
        sub = self.lc.subscribe("NOMINAL_COMMAND", self.handle_command)
        sub.set_queue_capacity(1)

        while True:
            if self.keyboard is not None and self.keyboard.should_terminate:
                self.publish_zero_velocity()
                print("Terminated by user.")
                break
            self.lc.handle_timeout(int(self.dt * 1000))


def main():
    parser = argparse.ArgumentParser(description="OCR Safety Filter Node")
    parser.add_argument(
        "--config",
        type=str,
        default="hardware/configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Start unpaused (for simulation)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup keyboard controller
    from utils.keyboard_controller import KeyboardController
    keyboard = KeyboardController(is_paused=not args.no_pause)
    keyboard.start()

    try:
        node = SafetyFilterNode(config, keyboard=keyboard)
        node.run()
    finally:
        keyboard.stop()


if __name__ == "__main__":
    main()
