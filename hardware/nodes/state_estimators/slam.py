#!/usr/bin/env python3
"""SLAM-based state estimator node using BreezySLAM.

Subscribes to raw LiDAR scans and publishes pose estimates and occupancy maps.

LCM Channels:
    Subscribes: LIDAR_SCAN
    Publishes: ROBOT_STATE, OCCUPANCY_MAP

Usage:
    python -m hardware.nodes.state_estimators.slam --config hardware/configs/default.yaml
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy import interpolate
from skimage.measure import block_reduce

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import lcm
except ImportError:
    raise ImportError(
        "LCM not installed. Install with: pip install lcm\n"
        "See https://lcm-proj.github.io/ for more information."
    )

try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import RPLidarA1 as LaserModel
except ImportError:
    raise ImportError(
        "BreezySLAM not installed. Install with: pip install -e libraries/BreezySLAM/python"
    )

from hardware.lcm_types import lidar_scan, robot_state, occupancy_map
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class SLAMNode:
    """SLAM node for state estimation and occupancy mapping."""

    def __init__(self, config: dict):
        self.config = config

        # Load training options for LiDAR offset
        with open(config["model"]["options"], "rb") as f:
            options = pickle.load(f)
        self.rel_lidar_position = options.get("rel_lidar_position", [0.2, 0])

        # SLAM parameters (all from config)
        self.map_size_pixels = config["slam"]["map_size_pixels"]
        self.map_size_meters = config["slam"]["map_size_meters"]
        self.reduced_map_size = config["slam"]["reduced_map_size"]
        self.quality_threshold = config["lidar"]["quality_threshold"]
        self.occupancy_threshold = config["slam"]["occupancy_threshold"]
        self.empty_threshold = config["slam"]["empty_threshold"]
        self.interp_radius = config["slam"]["interp_radius"]
        self.map_publish_period = config["slam"]["map_publish_period"]

        # Check LiDAR offset
        if self.rel_lidar_position[1] != 0:
            print("Warning: BreezySLAM cannot handle a LiDAR offset in the y direction")

        # Initialize SLAM
        # Offset is from robot center to LiDAR in mm (positive = LiDAR in front)
        lidar_offset_mm = -1000 * self.rel_lidar_position[0]
        self.slam = RMHC_SLAM(
            LaserModel(offsetMillimeters=lidar_offset_mm),
            self.map_size_pixels,
            self.map_size_meters,
        )

        # Map storage
        self.mapbytes = bytearray(self.map_size_pixels * self.map_size_pixels)
        self.prev_lidar_map = np.zeros((self.map_size_pixels, self.map_size_pixels))
        self.prev_occupancy_map = np.zeros(
            (self.map_size_pixels, self.map_size_pixels), dtype=bool
        )

        # Get initial position (used as origin)
        self.ox, self.oy, self.otheta = self.slam.getpos()

        # Timing
        self.last_map_publish_time = time.time()

        # LCM
        self.lc = lcm.LCM()

        print(f"SLAM node initialized with {self.map_size_pixels}x{self.map_size_pixels} map")
        print(f"LiDAR offset: {self.rel_lidar_position} m")

    def handle_lidar(self, channel, data):
        """Process incoming LiDAR scan."""
        scan = lidar_scan.decode(data)
        count = scan.count
        if count == 0:
            return

        # Extract valid points
        angles = np.array(scan.angle[:count])
        ranges = np.array(scan.range[:count])
        quality = np.array(scan.quality[:count])

        # Convert angles to degrees for SLAM (0-360)
        # rplidar.py already negates angles (CW -> CCW), so just convert to degrees
        angles_deg = np.degrees(angles) % 360

        # Filter by quality
        valid_mask = quality > self.quality_threshold
        if not np.any(valid_mask):
            return

        valid_angles = angles_deg[valid_mask]
        valid_ranges = ranges[valid_mask] * 1000  # Convert to mm for SLAM

        # Sort by angle
        sorted_indices = np.argsort(valid_angles)
        sorted_angles = valid_angles[sorted_indices]
        sorted_ranges = valid_ranges[sorted_indices]

        # Interpolate to 360 degrees
        interp_angles = np.arange(360)
        interp_func = interpolate.interp1d(
            sorted_angles,
            sorted_ranges,
            kind="nearest",
            bounds_error=False,
            fill_value=(sorted_ranges[0], sorted_ranges[-1]),
        )
        interp_ranges = interp_func(interp_angles)

        # Mark points with no nearby measurements as invalid
        angle_dists = np.abs(
            (interp_angles[:, np.newaxis] - sorted_angles + 180) % 360 - 180
        )
        interp_ranges[
            angle_dists.min(axis=-1) > np.degrees(self.interp_radius)
        ] = float("inf")

        # Update SLAM
        self.slam.update(list(interp_ranges), scan_angles_degrees=list(interp_angles))

        # Get pose estimate
        mx, my, theta_deg = self.slam.getpos()

        # Convert to robot frame (relative to start position)
        x = (self.ox - mx) / 1000.0  # mm to m
        y = (self.oy - my) / 1000.0

        # Convert heading to radians, wrap to [-pi, pi]
        theta = np.radians(theta_deg % 360)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Publish state estimate
        state_msg = robot_state()
        state_msg.timestamp = scan.timestamp
        state_msg.x = x
        state_msg.y = y
        state_msg.theta = theta
        self.lc.publish("ROBOT_STATE", state_msg.encode())

        # Publish occupancy map periodically
        current_time = time.time()
        if current_time - self.last_map_publish_time > self.map_publish_period:
            self.last_map_publish_time = current_time
            self._publish_occupancy_map()

    def _publish_occupancy_map(self):
        """Extract and publish occupancy grid."""
        self.slam.getmap(self.mapbytes)

        # Convert to numpy array (transpose and flip to match world coordinates)
        raw_map = (
            np.array(self.mapbytes)
            .reshape(self.map_size_pixels, self.map_size_pixels)
            .T[::-1, ::-1]
        )
        self.prev_lidar_map = 1 - raw_map / 255.0

        # Update occupancy with hysteresis
        self.prev_occupancy_map = np.logical_or(
            self.prev_occupancy_map, self.prev_lidar_map > self.occupancy_threshold
        )
        self.prev_occupancy_map = np.logical_and(
            self.prev_occupancy_map, self.prev_lidar_map > self.empty_threshold
        )

        # Reduce map size for planner
        reduction_factor = self.map_size_pixels // self.reduced_map_size
        reduced_map = block_reduce(
            self.prev_occupancy_map, block_size=reduction_factor, func=np.max
        )

        # Publish
        map_msg = occupancy_map()
        map_msg.timestamp = int(time.time() * 1e6)
        map_msg.width = self.reduced_map_size
        map_msg.height = self.reduced_map_size
        map_msg.resolution = self.map_size_meters / self.reduced_map_size
        map_msg.origin_x = -self.map_size_meters / 2
        map_msg.origin_y = -self.map_size_meters / 2

        # Flatten and pad to max size
        flat_map = reduced_map.flatten().tolist()
        map_msg.map = flat_map + [False] * (40000 - len(flat_map))

        self.lc.publish("OCCUPANCY_MAP", map_msg.encode())

    def run(self):
        """Main loop."""
        print("SLAM node started.")
        print("Subscribed to: LIDAR_SCAN")
        print("Publishing to: ROBOT_STATE, OCCUPANCY_MAP")

        subscription = self.lc.subscribe("LIDAR_SCAN", self.handle_lidar)
        subscription.set_queue_capacity(1)

        try:
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            print("\nSLAM node stopped.")


def main():
    parser = argparse.ArgumentParser(description="SLAM State Estimator Node")
    parser.add_argument(
        "--config",
        type=str,
        default="hardware/configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    node = SLAMNode(config)
    node.run()


if __name__ == "__main__":
    main()
