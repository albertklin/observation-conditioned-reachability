#!/usr/bin/env python3
"""Simulated robot node.

Simulates robot dynamics, LiDAR sensing, and state estimation.
Subscribes to SAFE_COMMAND and steps the simulation.

LCM Channels:
    Subscribes: SAFE_COMMAND
    Publishes: LIDAR_SCAN, ROBOT_STATE, OCCUPANCY_MAP

Usage:
    python -m hardware.nodes.robots.simulated --config hardware/configs/simulation.yaml
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import yaml
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
    from breezyslam.sensors import RPLidarA1 as LaserModel  # RPLidarA1 same as A2 for sim
except ImportError:
    raise ImportError(
        "BreezySLAM not installed. Install with: pip install -e libraries/BreezySLAM/python"
    )

from hardware.lcm_types import lidar_scan, robot_state, velocity_command, occupancy_map

from utils.dynamics import Dubins3D


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_environment(env_path: str):
    """Load environment from pickle file."""
    with open(env_path, "rb") as f:
        return pickle.load(f)


class SimulatedRobotNode:
    """Simulated robot node with LiDAR and state publishing."""

    def __init__(self, config: dict):
        self.config = config
        sim_config = config.get("simulation", {})

        # Load environment
        env_path = sim_config.get("environment")
        if env_path:
            print(f"Loading environment from {env_path}...")
            self.env = load_environment(env_path)
        else:
            # Create default environment with boundary walls and central obstacle
            print("Creating default environment...")
            from utils.simulation_utils.environment import Environment
            from utils.simulation_utils.obstacle import BoxObstacle, CircularObstacle
            # 14m x 10m room from (-2,-5) to (12,5) — matches eval boundary walls
            t = 0.1  # wall thickness
            obstacles = [
                BoxObstacle(center=np.array([5.0, 5.0]), length=14.0, width=t, height=1.0, angle=0),           # top wall
                BoxObstacle(center=np.array([5.0, -5.0]), length=14.0, width=t, height=1.0, angle=0),          # bottom wall
                BoxObstacle(center=np.array([-2.0, 0.0]), length=10.0, width=t, height=1.0, angle=np.pi / 2),  # left wall
                BoxObstacle(center=np.array([12.0, 0.0]), length=10.0, width=t, height=1.0, angle=np.pi / 2),  # right wall
                CircularObstacle(center=np.array([5.0, 0.0]), radius=1.0, height=1.0),                         # central obstacle
            ]
            self.env = Environment(obstacles=obstacles)

        # Load training options
        with open(config["model"]["options"], "rb") as f:
            options = pickle.load(f)
        self.rel_lidar_position = options.get("rel_lidar_position", [0.2, 0])

        # Dynamics
        self.dynamics = Dubins3D()
        self.dt = config["robot"]["dt"]

        # Robot state
        initial_state = sim_config.get("initial_state", [0.0, 0.0, 0.0])
        self.state = np.array(initial_state, dtype=np.float64)

        # Disturbance for simulation (constant bias, properly integrated via dynamics)
        disturbance_xy = sim_config.get("disturbance_xy", 0.1)
        disturbance_th = sim_config.get("disturbance_th", 0.1)
        bias_angle = np.random.uniform(0, 2 * np.pi)
        bias_sign_th = np.random.choice([-1, 1])
        self.dynamics.set_bias(
            disturbance_xy * np.cos(bias_angle),
            disturbance_xy * np.sin(bias_angle),
            disturbance_th * bias_sign_th,
        )
        print(f"Disturbance bias: xy=({disturbance_xy * np.cos(bias_angle):.3f}, "
              f"{disturbance_xy * np.sin(bias_angle):.3f}), "
              f"th={disturbance_th * bias_sign_th:.3f}")

        # LiDAR parameters
        self.lidar_min = config["lidar"]["min_range"]
        self.lidar_max = config["lidar"]["max_range"]
        self.quality_threshold = config["lidar"]["quality_threshold"]
        self.count_range = (600, 1200)
        self.quality_range = (39, 49)

        # SLAM parameters (matching run_sims.py)
        self.slam_map_size_pixels = config["slam"]["map_size_pixels"]
        self.slam_map_size_meters = config["slam"]["map_size_meters"]
        self.slam_reduced_map_size = config["slam"]["reduced_map_size"]
        self.slam_occupancy_threshold = config["slam"]["occupancy_threshold"]
        self.slam_empty_threshold = config["slam"]["empty_threshold"]
        self.map_publish_period = config["slam"]["map_publish_period"]

        # Initialize SLAM (matching run_sims.py)
        lidar_offset_mm = -1000 * self.rel_lidar_position[0]
        self.slam = RMHC_SLAM(
            LaserModel(offsetMillimeters=lidar_offset_mm),
            self.slam_map_size_pixels,
            self.slam_map_size_meters,
        )
        self.slam_mapbytes = bytearray(self.slam_map_size_pixels * self.slam_map_size_pixels)
        self.slam_prev_occupancy_map = np.zeros(
            (self.slam_map_size_pixels, self.slam_map_size_pixels), dtype=bool
        )

        # Timing
        self.last_command_time = time.time()
        self.last_map_publish_time = time.time()
        self.last_control = np.array([0.0, 0.0])

        # LCM
        self.lc = lcm.LCM()

        # Goal tracking
        self.goal_position = np.array(sim_config.get("goal", [10.0, 0.0]))
        self.goal_radius = 0.5
        self.goal_reached = False

        print(f"Simulated robot initialized at {self.state}")
        print(f"Goal: {self.goal_position}")

    def compute_lidar_scan(self) -> lidar_scan:
        """Compute simulated LiDAR scan."""
        # Compute LiDAR position
        cth, sth = np.cos(self.state[2]), np.sin(self.state[2])
        lidar_pos = self.state[:2].copy()
        lidar_pos[0] += self.rel_lidar_position[0] * cth - self.rel_lidar_position[1] * sth
        lidar_pos[1] += self.rel_lidar_position[0] * sth + self.rel_lidar_position[1] * cth

        # Generate random angles (matching run_sims.py: raw_angle = np.random.uniform(-np.pi, np.pi, 1400))
        count = np.random.randint(*self.count_range)
        angles = np.random.uniform(-np.pi, np.pi, 1400)

        # Raycast against environment
        ranges = self.env.read_lidar(
            lidar_pos[np.newaxis],
            (angles + self.state[2])[np.newaxis],
            min_distance=self.lidar_min,
            max_distance=self.lidar_max,
        )[0]

        # Generate quality values
        qualities = np.random.uniform(*self.quality_range, 1400)
        qualities[count:] = 0  # Zero out beyond count

        # Create message
        msg = lidar_scan()
        msg.timestamp = int(time.time() * 1e6)
        msg.count = count

        for i in range(1400):
            msg.angle[i] = angles[i]
            msg.range[i] = ranges[i]
            msg.quality[i] = int(qualities[i])

        return msg, angles, ranges, qualities, count

    def update_slam(self, angles: np.ndarray, ranges: np.ndarray, qualities: np.ndarray, count: int):
        """Update SLAM with new scan.

        Matches run_sims.py exactly (lines 302-306):
            slam_interp_angle_deg = np.arange(360)
            slam_interp_angle_rad = ((slam_interp_angle_deg*np.pi/180)+np.pi)%(2*np.pi)-np.pi
            slam_interp_dist = interpolate.interp1d(sorted_raw_angle, sorted_raw_lidar, ...)(slam_interp_angle_rad)
            slam.update(list(1000*slam_interp_dist), scan_angles_degrees=list(slam_interp_angle_deg))

        Args:
            angles: LiDAR angles in robot frame (radians)
            ranges: LiDAR ranges (meters)
            qualities: Quality values per ray
            count: Number of valid rays in this scan
        """
        # Filter by quality and count
        # Matching run_sims.py: is_valid = np.logical_and(raw_quality > quality_threshold, np.arange(1400) < count)
        valid_mask = np.logical_and(qualities > self.quality_threshold, np.arange(len(angles)) < count)
        valid_angles = angles[valid_mask]  # Keep in radians
        valid_ranges = ranges[valid_mask]

        if len(valid_angles) == 0:
            return

        # Sort by angle (in radians)
        # Matching run_sims.py: sorted_indices = np.argsort(valid_raw_angle)
        sorted_indices = np.argsort(valid_angles)
        sorted_angles = valid_angles[sorted_indices]
        sorted_ranges = valid_ranges[sorted_indices]

        # Interpolate to 360-degree scan for SLAM
        # Matching run_sims.py exactly:
        #   slam_interp_angle_deg = np.arange(360)
        #   slam_interp_angle_rad = ((slam_interp_angle_deg*np.pi/180)+np.pi)%(2*np.pi)-np.pi
        #   slam_interp_dist = interpolate.interp1d(sorted_raw_angle, sorted_raw_lidar, ...)(slam_interp_angle_rad)
        slam_interp_angle_deg = np.arange(360)
        slam_interp_angle_rad = ((slam_interp_angle_deg * np.pi / 180) + np.pi) % (2 * np.pi) - np.pi

        interp_fn = interpolate.interp1d(
            sorted_angles,
            sorted_ranges,
            kind="nearest",
            bounds_error=False,
            fill_value=(sorted_ranges[0], sorted_ranges[-1]),
        )
        slam_interp_dist = interp_fn(slam_interp_angle_rad)

        # Update SLAM (ranges in mm)
        # Matching run_sims.py: slam.update(list(1000*slam_interp_dist), scan_angles_degrees=list(slam_interp_angle_deg))
        self.slam.update(
            list((1000 * slam_interp_dist).astype(np.float64)),
            scan_angles_degrees=list(slam_interp_angle_deg),
        )

    def publish_occupancy_map(self):
        """Publish occupancy map.

        Matches run_sims.py exactly:
            slam.getmap(slam_mapbytes)
            slam_prev_lidar_map = 1 - np.array(slam_mapbytes).reshape(...).T[::-1, ::-1]/255
            slam_prev_occupancy_map = np.logical_or(slam_prev_occupancy_map, slam_prev_lidar_map > SLAM_OCCUPANCY_THRESHOLD)
            slam_prev_occupancy_map = np.logical_and(slam_prev_occupancy_map, slam_prev_lidar_map > SLAM_EMPTY_THRESHOLD)
            slam_map = block_reduce(slam_prev_occupancy_map, block_size=..., func=np.max)
        """
        self.slam.getmap(self.slam_mapbytes)

        # Convert to numpy array
        slam_prev_lidar_map = (
            1 - np.array(self.slam_mapbytes)
            .reshape(self.slam_map_size_pixels, self.slam_map_size_pixels)
            .T[::-1, ::-1] / 255
        )

        # Update occupancy with hysteresis
        self.slam_prev_occupancy_map = np.logical_or(
            self.slam_prev_occupancy_map, slam_prev_lidar_map > self.slam_occupancy_threshold
        )
        self.slam_prev_occupancy_map = np.logical_and(
            self.slam_prev_occupancy_map, slam_prev_lidar_map > self.slam_empty_threshold
        )

        # Reduce map size
        reduction_factor = self.slam_map_size_pixels // self.slam_reduced_map_size
        slam_map = block_reduce(
            self.slam_prev_occupancy_map, block_size=reduction_factor, func=np.max
        )

        # Publish
        map_msg = occupancy_map()
        map_msg.timestamp = int(time.time() * 1e6)
        map_msg.width = self.slam_reduced_map_size
        map_msg.height = self.slam_reduced_map_size
        map_msg.resolution = self.slam_map_size_meters / self.slam_reduced_map_size
        map_msg.origin_x = -self.slam_map_size_meters / 2
        map_msg.origin_y = -self.slam_map_size_meters / 2

        flat_map = slam_map.flatten().tolist()
        map_msg.map = flat_map + [False] * (40000 - len(flat_map))

        self.lc.publish("OCCUPANCY_MAP", map_msg.encode())

    def handle_command(self, channel, data):
        """Handle incoming velocity command and step simulation."""
        cmd = velocity_command.decode(data)
        current_time = time.time()

        # Get control
        control = np.array([cmd.v_x, cmd.v_yaw])

        # Compute actual dt
        dt = current_time - self.last_command_time
        if dt <= 0 or dt > 1.0:  # Sanity check
            dt = self.dt

        # Step dynamics (disturbance bias is integrated via dynamics.set_bias)
        self.state = self.dynamics.runge_kutta_step(
            self.state[np.newaxis],
            control[np.newaxis],
            dt,
        )[0]
        self.state = self.dynamics.wrap_states(self.state[np.newaxis])[0]

        self.last_command_time = current_time
        self.last_control = control

    def run(self):
        """Main loop."""
        print("Simulated robot started.")
        print("Subscribed to: SAFE_COMMAND")
        print("Publishing to: LIDAR_SCAN, ROBOT_STATE, OCCUPANCY_MAP")

        sub = self.lc.subscribe("SAFE_COMMAND", self.handle_command)
        sub.set_queue_capacity(1)

        try:
            while True:
                # Compute and publish LiDAR scan
                scan_msg, angles, ranges, qualities, count = self.compute_lidar_scan()
                self.lc.publish("LIDAR_SCAN", scan_msg.encode())

                # Update SLAM (no odometry, matching original run_sims.py)
                self.update_slam(angles, ranges, qualities, count)

                # Publish state (ground truth for simulation)
                state_msg = robot_state()
                state_msg.timestamp = scan_msg.timestamp
                state_msg.x = self.state[0]
                state_msg.y = self.state[1]
                state_msg.theta = self.state[2]
                self.lc.publish("ROBOT_STATE", state_msg.encode())

                # Publish occupancy map periodically
                current_time = time.time()
                if current_time - self.last_map_publish_time > self.map_publish_period:
                    self.last_map_publish_time = current_time
                    self.publish_occupancy_map()

                # Check goal
                dist_to_goal = np.linalg.norm(self.state[:2] - self.goal_position)

                # Track goal reached state (print once)
                if dist_to_goal < self.goal_radius and not self.goal_reached:
                    self.goal_reached = True
                    print(f"\nGoal reached! Infrastructure continues running.")
                elif dist_to_goal >= self.goal_radius:
                    self.goal_reached = False

                # Print status
                status = "AT GOAL" if self.goal_reached else f"dist: {dist_to_goal:.2f}m"
                print(
                    f"\rState: ({self.state[0]:5.2f}, {self.state[1]:5.2f}, {self.state[2]:5.2f}) | "
                    f"Cmd: ({self.last_control[0]:.2f}, {self.last_control[1]:.2f}) | "
                    f"{status}",
                    end="",
                )

                # Handle incoming commands (non-blocking)
                self.lc.handle_timeout(int(self.dt * 1000))

        except KeyboardInterrupt:
            print("\nSimulated robot stopped.")


def main():
    parser = argparse.ArgumentParser(description="Simulated Robot Node")
    parser.add_argument(
        "--config",
        type=str,
        default="hardware/configs/simulation.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    node = SimulatedRobotNode(config)
    node.run()


if __name__ == "__main__":
    main()
