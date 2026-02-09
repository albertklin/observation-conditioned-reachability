#!/usr/bin/env python3
"""Model Predictive Sampling (MPS) planner node.

Computes nominal velocity commands using model predictive sampling.
Subscribes to state estimates and occupancy maps, publishes nominal commands.

LCM Channels:
    Subscribes: ROBOT_STATE, OCCUPANCY_MAP
    Publishes: NOMINAL_COMMAND, MPS_TRAJECTORY

Usage:
    python -m hardware.nodes.planners.mps --config hardware/configs/default.yaml --goal-x 10 --goal-y 0
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    import lcm
except ImportError:
    raise ImportError(
        "LCM not installed. Install with: pip install lcm\n"
        "See https://lcm-proj.github.io/ for more information."
    )

from hardware.lcm_types import robot_state, velocity_command, occupancy_map, mps_trajectory

from utils.dynamics import Dubins3D
from utils.navigation_task import NavigationTask
from utils.predictive_sampler import PredictiveSampler
from utils.online_environment import OnlineEnvironment


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MPSPlannerNode:
    """Model Predictive Sampling planner node."""

    def __init__(self, config: dict, goal_position: np.ndarray = None):
        self.config = config

        # Goal position (can be updated dynamically)
        if goal_position is None:
            goal_position = np.array([10.0, 0.0])  # Default goal
        self.goal_position = goal_position
        self.goal_radius = 0.5

        # Control bounds
        self.control_min = np.array([
            config["control"]["v_x_min"],
            config["control"]["v_yaw_min"],
        ])
        self.control_max = np.array([
            config["control"]["v_x_max"],
            config["control"]["v_yaw_max"],
        ])

        # Planner parameters
        self.dt = config["planner"].get("dt", 0.2)
        self.horizon = config["planner"]["horizon"]
        self.num_steps = int(self.horizon / self.dt)
        self.num_samples = config["planner"]["num_samples"]
        noise_scale_factor = config["planner"]["noise_scale_factor"]
        self.noise_scale = (self.control_max - self.control_min) * noise_scale_factor

        # Map parameters
        self.map_size_meters = config["slam"]["map_size_meters"]
        self.reduced_map_size = config["slam"]["reduced_map_size"]

        # Setup environment for planning
        map_X = np.linspace(
            -self.map_size_meters / 2, self.map_size_meters / 2, self.reduced_map_size
        )
        map_Y = np.linspace(
            -self.map_size_meters / 2, self.map_size_meters / 2, self.reduced_map_size
        )
        self.environment = OnlineEnvironment(
            X=map_X, Y=map_Y, TH=np.array([-np.pi, np.pi])
        )

        # Setup dynamics and task
        self.dynamics = Dubins3D()
        self.task = NavigationTask(
            robot_radius=config["robot"]["radius"],
            goal_position=self.goal_position,
            goal_radius=self.goal_radius,
            environment=self.environment,
            dynamics=self.dynamics,
        )

        # Setup predictive sampler
        self.predictive_sampler = PredictiveSampler(
            self.task,
            self.num_samples,
            self.control_min,
            self.control_max,
            self.noise_scale,
            self.dt,
        )

        # Warm start control sequence
        control_center = (self.control_max + self.control_min) / 2
        self.nominal_control_sequence = np.tile(control_center, (self.num_steps, 1))

        # LCM
        self.lc = lcm.LCM()

        # State tracking
        self.latest_state = None
        self.last_publish_time = time.time()
        self._map_updated = False

        print(f"MPS planner initialized with {self.num_samples} samples, {self.horizon}s horizon")
        print(f"Goal: {self.goal_position}")

    def handle_state(self, channel, data):
        """Process incoming state estimate and compute control."""
        state_msg = robot_state.decode(data)
        current_time = time.time()

        state = np.array([state_msg.x, state_msg.y, state_msg.theta])
        self.latest_state = state

        # Check if we've reached the goal
        dist_to_goal = np.linalg.norm(state[:2] - self.goal_position)
        if dist_to_goal < self.goal_radius:
            # At goal, publish zero velocity
            cmd = velocity_command()
            cmd.timestamp = state_msg.timestamp
            cmd.v_x = 0.0
            cmd.v_yaw = 0.0
            cmd.is_safe = False
            self.lc.publish("NOMINAL_COMMAND", cmd.encode())
            return

        # Compute optimal control sequence
        optimal_control_seq = self.predictive_sampler.optimal_control_sequence(
            state, self.nominal_control_sequence,
            recompute_shortest_paths_cost_grid=self._map_updated,
        )
        self._map_updated = False

        # Publish nominal command (first control in sequence)
        cmd = velocity_command()
        cmd.timestamp = state_msg.timestamp
        cmd.v_x = float(optimal_control_seq[0, 0])
        cmd.v_yaw = float(optimal_control_seq[0, 1])
        cmd.is_safe = False  # Will be set by safety filter
        self.lc.publish("NOMINAL_COMMAND", cmd.encode())

        # Compute and publish trajectory for visualization
        traj_msg = mps_trajectory()
        traj_msg.timestamp = state_msg.timestamp
        traj_msg.num_points = min(len(optimal_control_seq), 100)

        # Roll out trajectory
        traj_state = state.copy()
        for i in range(traj_msg.num_points):
            traj_msg.xs[i] = traj_state[0]
            traj_msg.ys[i] = traj_state[1]
            traj_msg.ths[i] = traj_state[2]
            traj_state = self.dynamics.runge_kutta_step(
                states=traj_state[None],
                controls=optimal_control_seq[None, i],
                timesteps=self.dt,
            )[0]

        self.lc.publish("MPS_TRAJECTORY", traj_msg.encode())

        # Update warm start for next iteration
        self.nominal_control_sequence[:-1] = optimal_control_seq[1:]
        self.nominal_control_sequence[-1] = (self.control_max + self.control_min) / 2

        # Print status
        publish_latency = current_time - self.last_publish_time
        self.last_publish_time = current_time
        print(
            f"\rState: ({state[0]:5.2f}, {state[1]:5.2f}, {state[2]:5.2f}) | "
            f"Cmd: ({cmd.v_x:.2f}, {cmd.v_yaw:.2f}) | "
            f"Goal dist: {dist_to_goal:.2f}m | "
            f"dt: {publish_latency:.3f}s",
            end="",
        )

    def handle_occupancy_map(self, channel, data):
        """Process incoming occupancy map."""
        map_msg = occupancy_map.decode(data)

        # Extract occupancy grid
        width = map_msg.width
        height = map_msg.height
        flat_map = np.array(map_msg.map[: width * height], dtype=bool)
        grid = flat_map.reshape(width, height)

        # Update environment (expand to 3D grid for TH dimension)
        grid_3d = np.stack([grid, grid], axis=-1)
        self.environment.set_occupancy_grid(grid_3d)
        self._map_updated = True

    def set_goal(self, goal_position: np.ndarray):
        """Update the goal position."""
        self.goal_position = goal_position
        self.task = NavigationTask(
            robot_radius=self.config["robot"]["radius"],
            goal_position=self.goal_position,
            goal_radius=self.goal_radius,
            environment=self.environment,
            dynamics=self.dynamics,
        )
        self.predictive_sampler = PredictiveSampler(
            self.task,
            self.num_samples,
            self.control_min,
            self.control_max,
            self.noise_scale,
            self.dt,
        )
        print(f"\nGoal updated to: {goal_position}")

    def run(self):
        """Main loop."""
        print("MPS planner started.")
        print("Subscribed to: ROBOT_STATE, OCCUPANCY_MAP")
        print("Publishing to: NOMINAL_COMMAND, MPS_TRAJECTORY")

        sub = self.lc.subscribe("ROBOT_STATE", self.handle_state)
        sub.set_queue_capacity(1)

        sub = self.lc.subscribe("OCCUPANCY_MAP", self.handle_occupancy_map)
        sub.set_queue_capacity(1)

        try:
            while True:
                self.lc.handle()
        except KeyboardInterrupt:
            print("\nMPS planner stopped.")


def main():
    parser = argparse.ArgumentParser(description="MPS Planner Node")
    parser.add_argument(
        "--config",
        type=str,
        default="hardware/configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--goal-x",
        type=float,
        default=10.0,
        help="Goal X position (default: 10.0)",
    )
    parser.add_argument(
        "--goal-y",
        type=float,
        default=0.0,
        help="Goal Y position (default: 0.0)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    goal = np.array([args.goal_x, args.goal_y])
    node = MPSPlannerNode(config, goal_position=goal)
    node.run()


if __name__ == "__main__":
    main()
