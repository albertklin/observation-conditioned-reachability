#!/usr/bin/env python3
"""Real-time visualization node for safety filter monitoring.

Subscribes to LCM channels and displays real-time plots of:
- Robot state and LiDAR scan
- Safety filter metrics (value, controls, disturbance)
- MPS planner trajectory and occupancy map

LCM Channels:
    Subscribes: LIDAR_SCAN, ROBOT_STATE, SAFETY_STATUS, MPS_TRAJECTORY, OCCUPANCY_MAP

Usage:
    python -m hardware.visualization --config hardware/configs/default.yaml
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lcm
except ImportError:
    raise ImportError(
        "LCM not installed. Install with: pip install lcm\n"
        "See https://lcm-proj.github.io/ for more information."
    )

try:
    from PyQt5.QtWidgets import QApplication
    import pyqtgraph as pg
except ImportError:
    raise ImportError(
        "PyQt5 and pyqtgraph are required for visualization.\n"
        "Install with: pip install pyqt5 pyqtgraph"
    )

from hardware.lcm_types import (
    lidar_scan,
    robot_state,
    safety_status,
    mps_trajectory,
    occupancy_map,
)
from utils.visualizations import StateLidarWindow, SafetyFilterWindow, MPSWindow


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class VisualizationNode:
    """LCM-based visualization node."""

    def __init__(self, config: dict):
        self.config = config

        # History for time-series plots
        self.history = {
            "times": [],
            "filter_times": [],  # Separate timestamps for safety filter data
            "values": [],
            "v_xs": [],
            "v_yaws": [],
            "dsts": [],
            "states": [],
            "raw_angles": [],
            "raw_lidars": [],
            "raw_qualities": [],
            "counts": [],
            "inp_lidars": [],
            "lidar_states": [],
        }

        # MPS info for planner visualization
        self.mps_info = {
            "occupancy_map": None,
            "optimal_state_sequence": None,
        }

        # Config values
        self.quality_threshold = config["lidar"]["quality_threshold"]
        self.lidar_max_range = config["lidar"]["max_range"]
        self.rel_lidar_position = config["lidar"].get("rel_position", [0.2, 0.0])
        self.control_min = [config["control"]["v_x_min"], config["control"]["v_yaw_min"]]
        self.control_max = [config["control"]["v_x_max"], config["control"]["v_yaw_max"]]
        self.value_threshold = config["filter"]["threshold"]
        self.map_size_pixels = config["slam"]["reduced_map_size"]
        self.map_size_meters = config["slam"]["map_size_meters"]

        # Latest state for LiDAR frame tracking
        self.latest_state = None
        self.latest_lidar_state = None

        # Thread lock for history updates
        self.lock = threading.Lock()

        # LCM
        self.lc = lcm.LCM()

    def handle_lidar(self, channel, data):
        """Handle incoming LiDAR scan."""
        scan = lidar_scan.decode(data)
        count = scan.count
        if count == 0:
            return

        with self.lock:
            self.history["raw_angles"].append(list(scan.angle[:count]))
            self.history["raw_lidars"].append(list(scan.range[:count]))
            self.history["raw_qualities"].append(list(scan.quality[:count]))
            self.history["counts"].append(count)

            # Track lidar state
            if self.latest_state is not None:
                self.latest_lidar_state = self.latest_state.copy()

    def handle_state(self, channel, data):
        """Handle incoming state estimate."""
        state_msg = robot_state.decode(data)
        state = np.array([state_msg.x, state_msg.y, state_msg.theta])

        with self.lock:
            self.latest_state = state
            self.history["states"].append(state.tolist())
            self.history["times"].append(time.time())

            if self.latest_lidar_state is not None:
                self.history["lidar_states"].append(self.latest_lidar_state.tolist())
            else:
                self.history["lidar_states"].append(state.tolist())

            # Trim history to last 1000 entries
            max_history = 1000
            if len(self.history["times"]) > max_history:
                for key in self.history:
                    if len(self.history[key]) > max_history:
                        self.history[key] = self.history[key][-max_history:]

    def handle_safety_status(self, channel, data):
        """Handle incoming safety status."""
        status = safety_status.decode(data)

        with self.lock:
            self.history["filter_times"].append(time.time())
            self.history["values"].append(status.value)
            self.history["v_xs"].append(status.safe_v_x)
            self.history["v_yaws"].append(status.safe_v_yaw)
            self.history["dsts"].append([status.dst_dxdy, status.dst_dth])
            self.history["inp_lidars"].append(list(status.inp_lidar[:status.num_rays]))

    def handle_trajectory(self, channel, data):
        """Handle incoming MPS trajectory."""
        traj = mps_trajectory.decode(data)
        n = traj.num_points

        with self.lock:
            self.mps_info["optimal_state_sequence"] = np.array([
                [traj.xs[i], traj.ys[i], traj.ths[i]] for i in range(n)
            ])

    def handle_occupancy_map(self, channel, data):
        """Handle incoming occupancy map."""
        map_msg = occupancy_map.decode(data)
        width = map_msg.width
        height = map_msg.height

        with self.lock:
            grid = np.array(map_msg.map[:width * height], dtype=float).reshape(width, height)
            self.mps_info["occupancy_map"] = grid

    def run_lcm_thread(self):
        """Run LCM message handling in background thread."""
        while True:
            self.lc.handle()

    def run(self):
        """Main loop - create windows and run Qt event loop."""
        print("Visualization node starting...")
        print("Subscribed to: LIDAR_SCAN, ROBOT_STATE, SAFETY_STATUS, MPS_TRAJECTORY, OCCUPANCY_MAP")

        # Subscribe to LCM channels
        sub = self.lc.subscribe("LIDAR_SCAN", self.handle_lidar)
        sub.set_queue_capacity(1)
        sub = self.lc.subscribe("ROBOT_STATE", self.handle_state)
        sub.set_queue_capacity(1)
        sub = self.lc.subscribe("SAFETY_STATUS", self.handle_safety_status)
        sub.set_queue_capacity(1)
        sub = self.lc.subscribe("MPS_TRAJECTORY", self.handle_trajectory)
        sub.set_queue_capacity(1)
        sub = self.lc.subscribe("OCCUPANCY_MAP", self.handle_occupancy_map)
        sub.set_queue_capacity(1)

        # Start LCM thread
        lcm_thread = threading.Thread(target=self.run_lcm_thread, daemon=True)
        lcm_thread.start()

        # Create Qt application
        app = QApplication([])

        # Create visualization windows
        state_window = StateLidarWindow(
            title="Robot State & LiDAR (World Frame)",
            update_period=0.1,
            vis_type="raw",
            plot_radius=5.0,
            rel_lidar_position=self.rel_lidar_position,
            quality_threshold=self.quality_threshold,
            max_range=self.lidar_max_range,
            time_span=10.0,
            history=self.history,
        )

        input_window = StateLidarWindow(
            title="Network Input (Ego Frame)",
            update_period=0.1,
            vis_type="input",
            plot_radius=5.0,
            rel_lidar_position=self.rel_lidar_position,
            quality_threshold=self.quality_threshold,
            max_range=self.lidar_max_range,
            time_span=10.0,
            history=self.history,
        )

        filter_window = SafetyFilterWindow(
            title="Safety Filter Monitor",
            update_period=0.1,
            control_min=self.control_min,
            control_max=self.control_max,
            value_threshold=self.value_threshold,
            time_span=10.0,
            history=self.history,
        )

        mps_window = MPSWindow(
            title="MPS Planner",
            update_period=0.2,
            mps_info=self.mps_info,
            history=self.history,
            map_size_pixels=self.map_size_pixels,
            map_size_meters=self.map_size_meters,
        )

        # Position windows (arranged for typical 1920x1080 or larger display)
        state_window.setGeometry(50, 50, 500, 500)
        input_window.setGeometry(570, 50, 500, 500)
        filter_window.setGeometry(1090, 50, 400, 700)
        mps_window.setGeometry(50, 570, 500, 500)

        # Show windows
        state_window.show()
        input_window.show()
        filter_window.show()
        mps_window.show()

        print("Visualization windows opened.")
        print("Close any window to exit.")

        # Run Qt event loop
        app.exec_()
        print("Visualization node stopped.")


def main():
    parser = argparse.ArgumentParser(description="Visualization Node")
    parser.add_argument(
        "--config",
        type=str,
        default="hardware/configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    node = VisualizationNode(config)
    node.run()


if __name__ == "__main__":
    main()
