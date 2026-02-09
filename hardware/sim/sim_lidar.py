"""Simulated LiDAR sensor."""

import numpy as np
from scipy import interpolate

from hardware.interfaces import Lidar


class SimLidar(Lidar):
    """Simulated LiDAR that raycasts against an environment.

    This simulates an RPLidar A2 sensor with configurable noise and quality
    characteristics. The default parameters match the RPLidar A2 used in
    the hardware experiments.
    """

    def __init__(
        self,
        environment,
        min_range: float = 0.2,
        max_range: float = 10.0,
        num_rays: int = 1400,
        count_range: tuple[int, int] = (600, 1200),
        quality_range: tuple[float, float] = (39, 49),
        quality_threshold: float = 40,
    ):
        """Initialize the simulated LiDAR.

        Args:
            environment: Environment object with read_lidar method.
            min_range: Minimum valid range in meters.
            max_range: Maximum valid range in meters.
            num_rays: Number of rays per scan.
            count_range: Range for random ray count (min, max).
            quality_range: Range for random quality values (min, max).
            quality_threshold: Minimum quality to consider a ray valid.
        """
        self._environment = environment
        self._min_range = min_range
        self._max_range = max_range
        self._num_rays = num_rays
        self._count_range = count_range
        self._quality_range = quality_range
        self._quality_threshold = quality_threshold

        # Current scan data
        self._ranges = np.zeros(num_rays)
        self._angles = np.random.uniform(-np.pi, np.pi, num_rays)
        self._qualities = np.zeros(num_rays)

    def get_scan(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Returns the latest raw LiDAR scan."""
        return self._ranges.copy(), self._angles.copy(), self._qualities.copy()

    def get_preprocessed_scan(
        self,
        thetas: np.ndarray,
        focus_radius: float,
        focus_fov: float,
        interp_radius: float = 0.1,
    ) -> np.ndarray:
        """Returns a preprocessed LiDAR scan ready for network input.

        Handles RPLidar A2-specific preprocessing:
        - Quality filtering (threshold-based)
        - Interpolation to target angles
        - Interpolation gap validation (interp_radius)
        - Range clipping
        - Focus masking

        Args:
            thetas: Target angles for interpolation (network input format).
            focus_radius: Ranges beyond min + focus_radius are clipped.
            focus_fov: Angles beyond focus_fov from forward are clipped.
            interp_radius: Max angular distance (rad) for valid interpolation.

        Returns:
            Preprocessed range array at the target angles.
        """
        ranges, angles, qualities = self.get_scan()

        # Filter by quality
        valid_mask = qualities > self._quality_threshold
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        if len(valid_angles) == 0:
            return np.full_like(thetas, self._max_range)

        # Sort by angle
        sorted_indices = np.argsort(valid_angles)
        sorted_angles = valid_angles[sorted_indices]
        sorted_ranges = valid_ranges[sorted_indices]

        # Interpolate to target angles
        interp_fn = interpolate.interp1d(
            sorted_angles,
            sorted_ranges,
            kind="nearest",
            bounds_error=False,
            fill_value=(sorted_ranges[0], sorted_ranges[-1]),
        )
        inp_lidar = interp_fn(thetas)

        # Invalidate interpolated points too far from any real measurement
        theta_abs_dists = np.abs(
            (thetas[:, np.newaxis] - sorted_angles + np.pi) % (2 * np.pi) - np.pi
        )
        inp_lidar[theta_abs_dists.min(axis=-1) > interp_radius] = float('inf')

        # Clip to valid range
        inp_lidar = np.clip(inp_lidar, self._min_range, self._max_range)

        # Apply focus mask (angles beyond FOV)
        inp_lidar[np.abs(thetas) > focus_fov] = self._max_range

        # Apply focus radius (ranges beyond nearest + radius)
        inp_lidar_min = np.min(inp_lidar)
        inp_lidar[inp_lidar > inp_lidar_min + focus_radius] = self._max_range

        return inp_lidar

    def update(self, position: np.ndarray, heading: float) -> None:
        """Updates the LiDAR scan from the current position.

        Args:
            position: LiDAR position [x, y] in world frame.
            heading: Robot heading in radians.
        """
        count = np.random.randint(self._count_range[0], self._count_range[1])
        self._angles = np.random.uniform(-np.pi, np.pi, self._num_rays)

        self._ranges = self._environment.read_lidar(
            position[np.newaxis],
            (self._angles + heading)[np.newaxis],
            min_distance=self._min_range,
            max_distance=self._max_range,
        )[0]

        self._qualities = np.random.uniform(
            self._quality_range[0], self._quality_range[1], self._num_rays
        )
        # Zero out qualities for rays beyond count
        self._qualities[count:] = 0

    @property
    def min_range(self) -> float:
        """Minimum valid range measurement in meters."""
        return self._min_range

    @property
    def max_range(self) -> float:
        """Maximum valid range measurement in meters."""
        return self._max_range


class IsaacGymLidar(Lidar):
    """LiDAR adapter for IsaacGym simulator.

    Wraps the IsaacGym simulator's raycasting to provide the Lidar interface.
    Simulates RPLidar A2 noise characteristics.
    """

    def __init__(
        self,
        simulator,
        min_range: float = 0.2,
        max_range: float = 10.0,
        num_rays: int = 1400,
        count_range: tuple[int, int] = (600, 1200),
        quality_range: tuple[float, float] = (39, 49),
        quality_threshold: float = 40,
    ):
        """Initialize the IsaacGym LiDAR adapter.

        Args:
            simulator: IsaacGymSimulator instance with read_lidar method.
            min_range: Minimum valid range in meters.
            max_range: Maximum valid range in meters.
            num_rays: Number of rays per scan.
            count_range: Range for random ray count (min, max).
            quality_range: Range for random quality values (min, max).
            quality_threshold: Minimum quality to consider a ray valid.
        """
        self._simulator = simulator
        self._min_range = min_range
        self._max_range = max_range
        self._num_rays = num_rays
        self._count_range = count_range
        self._quality_range = quality_range
        self._quality_threshold = quality_threshold

        # Current scan data
        self._ranges = np.zeros(num_rays)
        self._angles = np.random.uniform(-np.pi, np.pi, num_rays)
        self._qualities = np.zeros(num_rays)

    def get_scan(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Returns the latest raw LiDAR scan."""
        return self._ranges.copy(), self._angles.copy(), self._qualities.copy()

    def get_preprocessed_scan(
        self,
        thetas: np.ndarray,
        focus_radius: float,
        focus_fov: float,
        interp_radius: float = 0.1,
    ) -> np.ndarray:
        """Returns a preprocessed LiDAR scan ready for network input."""
        ranges, angles, qualities = self.get_scan()

        # Filter by quality
        valid_mask = qualities > self._quality_threshold
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        if len(valid_angles) == 0:
            return np.full_like(thetas, self._max_range)

        # Sort by angle
        sorted_indices = np.argsort(valid_angles)
        sorted_angles = valid_angles[sorted_indices]
        sorted_ranges = valid_ranges[sorted_indices]

        # Interpolate to target angles
        interp_fn = interpolate.interp1d(
            sorted_angles,
            sorted_ranges,
            kind="nearest",
            bounds_error=False,
            fill_value=(sorted_ranges[0], sorted_ranges[-1]),
        )
        inp_lidar = interp_fn(thetas)

        # Invalidate interpolated points too far from any real measurement
        theta_abs_dists = np.abs(
            (thetas[:, np.newaxis] - sorted_angles + np.pi) % (2 * np.pi) - np.pi
        )
        inp_lidar[theta_abs_dists.min(axis=-1) > interp_radius] = float('inf')

        # Clip to valid range
        inp_lidar = np.clip(inp_lidar, self._min_range, self._max_range)

        # Apply focus mask
        inp_lidar[np.abs(thetas) > focus_fov] = self._max_range

        # Apply focus radius
        inp_lidar_min = np.min(inp_lidar)
        inp_lidar[inp_lidar > inp_lidar_min + focus_radius] = self._max_range

        return inp_lidar

    def update(self, position: np.ndarray, heading: float) -> None:
        """Updates the LiDAR scan from the current position.

        Args:
            position: LiDAR position [x, y] in world frame.
            heading: Robot heading in radians.
        """
        count = np.random.randint(self._count_range[0], self._count_range[1])
        self._angles = np.random.uniform(-np.pi, np.pi, self._num_rays)

        self._ranges = self._simulator.read_lidar(
            position,
            self._angles + heading,
            min_distance=self._min_range,
            max_distance=self._max_range,
        )

        self._qualities = np.random.uniform(
            self._quality_range[0], self._quality_range[1], self._num_rays
        )
        self._qualities[count:] = 0

    @property
    def min_range(self) -> float:
        """Minimum valid range measurement in meters."""
        return self._min_range

    @property
    def max_range(self) -> float:
        """Maximum valid range measurement in meters."""
        return self._max_range
