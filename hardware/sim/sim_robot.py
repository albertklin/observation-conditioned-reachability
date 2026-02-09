"""Simulated robot using Dubins dynamics."""

import numpy as np

from utils.dynamics import Dubins3D


class SimRobot:
    """Simulated robot with Dubins dynamics.

    This robot simulates a differential-drive or unicycle robot
    using Dubins3D dynamics with configurable disturbances.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        dt: float = 0.05,
        disturbance_xy: float = 0.0,
        disturbance_th: float = 0.0,
    ):
        """Initialize the simulated robot.

        Args:
            initial_state: Initial robot state [x, y, theta].
            dt: Control timestep in seconds. Should match the expected
                control loop period (robot.dt in config). Note: this fixed dt
                is used for kinematic sim stepping only. The disturbance
                estimator in the safety filter uses wall-clock elapsed time
                (not this value) for its dynamics predictions.
            disturbance_xy: Maximum disturbance in x/y directions (m/s).
            disturbance_th: Maximum disturbance in theta (rad/s).
        """
        self._state = initial_state.copy()
        self._dt = dt
        self._disturbance_xy = disturbance_xy
        self._disturbance_th = disturbance_th
        self._v_x = 0.0
        self._v_yaw = 0.0

        self._dynamics = Dubins3D()

    def send_command(self, v_x: float, v_yaw: float) -> None:
        """Sends velocity command and steps the simulation.

        Args:
            v_x: Forward velocity in m/s.
            v_yaw: Yaw rate in rad/s.
        """
        self._v_x = v_x
        self._v_yaw = v_yaw

        # Apply disturbance
        if self._disturbance_xy > 0 or self._disturbance_th > 0:
            d_xy = np.random.uniform(-self._disturbance_xy, self._disturbance_xy, 2)
            d_th = np.random.uniform(-self._disturbance_th, self._disturbance_th)
            self._dynamics.set_bias(d_xy[0], d_xy[1], d_th)
        else:
            self._dynamics.set_bias(0, 0, 0)

        # Step dynamics
        control = np.array([[v_x, v_yaw]])
        self._state = self._dynamics.runge_kutta_step(
            self._state[np.newaxis], control, self._dt
        )[0]
        self._state = self._dynamics.wrap_states(self._state[np.newaxis])[0]

    def get_velocity(self) -> "tuple[float, float]":
        """Returns the current commanded velocity."""
        return self._v_x, self._v_yaw

    def get_state(self) -> np.ndarray:
        """Returns the current robot state [x, y, theta]."""
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Sets the robot state directly."""
        self._state = state.copy()

    @property
    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._dt
