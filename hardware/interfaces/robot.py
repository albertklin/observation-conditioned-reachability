"""Abstract interface for robot control."""

import abc
import numpy as np


class Robot(metaclass=abc.ABCMeta):
    """Abstract base class for robot control.

    Implementations should provide velocity control for differential-drive
    or Ackermann-style robots that can be modeled with Dubins dynamics.
    """

    @abc.abstractmethod
    def send_command(self, v_x: float, v_yaw: float) -> None:
        """Sends velocity command to the robot.

        Args:
            v_x: Forward velocity in m/s.
            v_yaw: Yaw rate in rad/s.
        """

    @abc.abstractmethod
    def get_velocity(self) -> "tuple[float, float]":
        """Returns the current robot velocity.

        Returns:
            Tuple of (v_x, v_yaw) where:
            - v_x: Forward velocity in m/s.
            - v_yaw: Yaw rate in rad/s.
        """

    @property
    @abc.abstractmethod
    def dt(self) -> float:
        """Control timestep in seconds."""
