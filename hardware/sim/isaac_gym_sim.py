"""IsaacGym-based realistic simulator for hardware testing.

This module provides a unified simulator that wraps the Walk-These-Ways
IsaacGym environment to provide realistic quadruped robot simulation.
"""

import os
import pickle
import numpy as np
import torch

# IsaacGym must be imported before torch
import isaacgym
assert isaacgym
from isaacgym import gymapi

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.walk_these_ways_utils.loaders import load_env
from utils.navigation_task import NavigationTask
from utils.dynamics import Dubins3D


class IsaacGymSimulator:
    """Unified simulator using IsaacGym for realistic robot physics.

    This simulator wraps the Walk-These-Ways environment to provide
    realistic quadruped robot dynamics, including:
    - Full rigid body dynamics
    - Actuator dynamics with latency
    - Ground contact and friction
    - Configurable payload and terrain friction

    The simulator provides methods for:
    - Stepping the simulation with velocity commands
    - Reading the robot state
    - Computing LiDAR scans via raycasting
    - Collision detection
    """

    def __init__(
        self,
        environment,
        task: NavigationTask,
        headless: bool = True,
        payload: float = 0.0,
        friction: float = 1.0,
        safe_color: tuple = (0.36, 0.5, 0.25),
        unsafe_color: tuple = (0.5, 0.25, 0.25),
    ):
        """Initialize the IsaacGym simulator.

        Args:
            environment: Environment object with obstacle data.
            task: NavigationTask defining goal and robot parameters.
            headless: If True, run without visualization.
            payload: Additional payload mass in kg.
            friction: Ground friction coefficient.
            safe_color: RGB color when robot is safe.
            unsafe_color: RGB color when robot is filtered.
        """
        self._environment = environment
        self._task = task
        self._headless = headless
        self._safe_color = safe_color
        self._unsafe_color = unsafe_color

        # Load the IsaacGym environment
        self._sim_env = load_env(
            "gait-conditioned-agility/pretrain-v0/train",
            task,
            payload=payload,
            friction=friction,
            headless=headless,
            body_color=safe_color,
        )

        # Set camera and lighting if not headless
        if not headless:
            self._sim_env.set_camera([5, 0, 11], [5, 1e-12, 0])
            self._sim_env.gym.set_light_parameters(
                self._sim_env.sim,
                0,
                gymapi.Vec3(0.8, 0.8, 0.8),
                gymapi.Vec3(0.8, 0.8, 0.8),
                gymapi.Vec3(-0.5, 0, -1),
            )
            self._sim_env.gym.set_light_parameters(
                self._sim_env.sim,
                1,
                gymapi.Vec3(0.8, 0.8, 0.8),
                gymapi.Vec3(0.8, 0.8, 0.8),
                gymapi.Vec3(0.5, 0, -1),
            )

        # Current observation (for stepping)
        self._obs = None

        # Dynamics model for state wrapping
        self._dynamics = Dubins3D()

    @property
    def dt(self) -> float:
        """Simulation timestep in seconds."""
        return self._sim_env.dt

    def reset(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        """Reset the simulation.

        Args:
            initial_state: Optional initial state [x, y, theta].
                          If None, uses origin.

        Returns:
            Initial robot state [x, y, theta].
        """
        self._obs = self._sim_env.reset()

        if initial_state is not None:
            self._sim_env.set_robot_state(
                x=initial_state[0], y=initial_state[1], th=initial_state[2]
            )
        else:
            self._sim_env.set_robot_state(x=0, y=0, th=0)

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get the current robot state.

        Returns:
            Robot state [x, y, theta] in world frame.
        """
        state = self._sim_env.current_dubins3d_state()
        return self._dynamics.wrap_states(state[np.newaxis])[0]

    def get_velocity(self) -> np.ndarray:
        """Get the current robot velocity.

        Returns:
            Robot velocity [v_x, v_y, v_yaw] in body frame.
        """
        return self._sim_env.base_lin_vel[0].cpu().numpy()

    def step(
        self,
        v_x: float,
        v_yaw: float,
        is_filtered: bool = False,
        add_line: bool = False,
    ) -> np.ndarray:
        """Step the simulation with velocity commands.

        Args:
            v_x: Forward velocity command in m/s.
            v_yaw: Yaw rate command in rad/s.
            is_filtered: If True, use unsafe color.
            add_line: If True, draw trajectory line.

        Returns:
            New robot state [x, y, theta].
        """
        body_color = self._unsafe_color if is_filtered else self._safe_color
        line_color = (0, 0, 0) if is_filtered else (0, 0, 1)

        self._obs = self._sim_env.step_dubins3d(
            v_x, v_yaw, self._obs, add_line=add_line, line_color=line_color, body_color=body_color
        )

        return self.get_state()

    def read_lidar(
        self,
        position: np.ndarray,
        angles: np.ndarray,
        min_distance: float = 0.2,
        max_distance: float = 12.0,
    ) -> np.ndarray:
        """Compute LiDAR scan via raycasting.

        Args:
            position: LiDAR position [x, y] in world frame.
            angles: Ray angles in world frame.
            min_distance: Minimum range in meters.
            max_distance: Maximum range in meters.

        Returns:
            Range measurements for each angle.
        """
        return self._environment.read_lidar(
            position[np.newaxis],
            angles[np.newaxis],
            min_distance=min_distance,
            max_distance=max_distance,
        )[0]

    def in_collision(self) -> bool:
        """Check if robot is in collision with obstacles.

        Returns:
            True if in collision.
        """
        return self._sim_env.in_collision()

    def refresh_drawings(self) -> None:
        """Refresh visualization drawings."""
        if not self._headless:
            self._sim_env.refresh_drawings()

    def remove_temp_lines(self) -> None:
        """Remove temporary visualization lines."""
        if not self._headless:
            self._sim_env.remove_temp_lines()

    def add_temp_line(
        self, x1: float, y1: float, x2: float, y2: float, color: tuple
    ) -> None:
        """Add a temporary line for visualization."""
        if not self._headless:
            self._sim_env.add_temp_line(x1, y1, x2, y2, color)

    def add_temp_box(
        self, x: float, y: float, th: float, length: float, width: float, color: tuple
    ) -> None:
        """Add a temporary box for visualization."""
        if not self._headless:
            self._sim_env.add_temp_box(x, y, th, length, width, color)

    def save_image(self, path: str) -> None:
        """Save current viewer image to file."""
        if not self._headless:
            self._sim_env.gym.write_viewer_image_to_file(self._sim_env.viewer, path)

    def close(self) -> None:
        """Close the simulator."""
        self._sim_env.close()


def load_environment(env_path: str):
    """Load an environment from a pickle file.

    Args:
        env_path: Path to environment pickle file.

    Returns:
        Environment object.
    """
    with open(env_path, "rb") as f:
        return pickle.load(f)
