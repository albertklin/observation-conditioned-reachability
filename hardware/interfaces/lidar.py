"""Abstract interface for LiDAR sensors."""

import abc
import numpy as np


class Lidar(metaclass=abc.ABCMeta):
    """Abstract base class for LiDAR sensors.

    The get_preprocessed_scan() method handles sensor-specific preprocessing
    to produce network-ready input. Different sensors may have different
    quality metrics, noise characteristics, and preprocessing requirements.
    """

    @abc.abstractmethod
    def get_scan(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Returns the latest raw LiDAR scan.

        Returns:
            Tuple of (ranges, angles, qualities) where:
            - ranges: Array of range measurements in meters.
            - angles: Array of corresponding angles in radians (ego-frame).
            - qualities: Array of quality/intensity values (sensor-specific).
        """

    @abc.abstractmethod
    def get_preprocessed_scan(
        self,
        thetas: np.ndarray,
        focus_radius: float,
        focus_fov: float,
        interp_radius: float = 0.1,
    ) -> np.ndarray:
        """Returns a preprocessed LiDAR scan ready for network input.

        This method handles sensor-specific preprocessing including:
        - Quality filtering
        - Interpolation to target angles
        - Interpolation gap validation (interp_radius)
        - Range clipping
        - Focus masking

        Args:
            thetas: Target angles for interpolation (network input format).
            focus_radius: Ranges beyond min + focus_radius are clipped.
            focus_fov: Angles beyond focus_fov from forward are clipped.
            interp_radius: Max angular distance (rad) for valid interpolation.
                Interpolated points farther than this from any real measurement
                are set to inf.

        Returns:
            Preprocessed range array at the target angles.
        """

    @property
    @abc.abstractmethod
    def min_range(self) -> float:
        """Minimum valid range measurement in meters."""

    @property
    @abc.abstractmethod
    def max_range(self) -> float:
        """Maximum valid range measurement in meters."""
