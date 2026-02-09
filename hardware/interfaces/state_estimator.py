"""Abstract interface for state estimation."""

import abc
import numpy as np


class StateEstimator(metaclass=abc.ABCMeta):
    """Abstract base class for state estimation.

    Implementations should provide robot pose estimates in a global frame.
    Common implementations include SLAM-based estimators, motion capture,
    or other localization systems.
    """

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """Returns the current estimated robot state.

        Returns:
            state: Robot state [x, y, theta] in the global frame.
        """

    @abc.abstractmethod
    def update(self, lidar_scan: np.ndarray, lidar_angles: np.ndarray) -> None:
        """Updates the state estimate with new sensor data.

        Args:
            lidar_scan: Array of range measurements in meters.
            lidar_angles: Array of corresponding angles in radians.
        """
