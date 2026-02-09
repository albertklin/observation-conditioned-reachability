"""Simulated state estimator using BreezySLAM."""

import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import RPLidarA1 as LaserModel
from skimage.measure import block_reduce

from hardware.interfaces import StateEstimator


class SimStateEstimator(StateEstimator):
    """SLAM-based state estimator using BreezySLAM.

    This estimator uses BreezySLAM for localization and mapping.
    It maintains an occupancy grid that can be used for planning.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        lidar_offset_x: float = 0.0,
        map_size_pixels: int = 2000,
        map_size_meters: float = 20.0,
        reduced_map_size: int = 200,
        occupancy_threshold: float = 0.85,
        empty_threshold: float = 0.01,
    ):
        """Initialize the SLAM-based state estimator.

        Args:
            initial_state: Initial robot state [x, y, theta].
            lidar_offset_x: LiDAR x offset from robot center in meters.
            map_size_pixels: SLAM map resolution in pixels.
            map_size_meters: SLAM map size in meters.
            reduced_map_size: Reduced map resolution for planning.
            occupancy_threshold: Threshold for marking cells as occupied.
            empty_threshold: Threshold for clearing cells.
        """
        self._state = initial_state.copy().astype(np.float64)
        self._lidar_offset_x = lidar_offset_x
        self._map_size_pixels = map_size_pixels
        self._map_size_meters = map_size_meters
        self._reduced_map_size = reduced_map_size
        self._occupancy_threshold = occupancy_threshold
        self._empty_threshold = empty_threshold

        # Initialize SLAM
        self._slam = RMHC_SLAM(
            LaserModel(offsetMillimeters=-1000 * lidar_offset_x),
            map_size_pixels,
            map_size_meters,
        )
        self._mapbytes = bytearray(map_size_pixels * map_size_pixels)
        self._prev_lidar_map = np.zeros((map_size_pixels, map_size_pixels))
        self._prev_occupancy_map = np.zeros((map_size_pixels, map_size_pixels), dtype=bool)
        self._occupancy_grid = None

    def get_state(self) -> np.ndarray:
        """Returns the current estimated robot state."""
        return self._state.copy()

    def update(self, lidar_scan: np.ndarray, lidar_angles: np.ndarray) -> None:
        """Updates the state estimate with new LiDAR data.

        Args:
            lidar_scan: Array of range measurements in meters.
            lidar_angles: Array of corresponding angles in radians.
        """
        # Sort by angle for interpolation
        sorted_indices = np.argsort(lidar_angles)
        sorted_angles = lidar_angles[sorted_indices]
        sorted_ranges = lidar_scan[sorted_indices]

        # Interpolate to 360-degree scan for SLAM
        slam_angles_deg = np.arange(360)
        slam_angles_rad = ((slam_angles_deg * np.pi / 180) + np.pi) % (2 * np.pi) - np.pi

        from scipy import interpolate
        interp_fn = interpolate.interp1d(
            sorted_angles,
            sorted_ranges,
            kind="nearest",
            bounds_error=False,
            fill_value=(sorted_ranges[0], sorted_ranges[-1]),
        )
        slam_ranges = interp_fn(slam_angles_rad)

        # Update SLAM
        self._slam.update(
            list((1000 * slam_ranges).astype(np.float64)),
            scan_angles_degrees=list(slam_angles_deg),
        )

        # Update map
        self._slam.getmap(self._mapbytes)
        lidar_map = (
            1
            - np.array(self._mapbytes).reshape(
                self._map_size_pixels, self._map_size_pixels
            ).T[::-1, ::-1]
            / 255
        )
        self._prev_occupancy_map = np.logical_or(
            self._prev_occupancy_map, lidar_map > self._occupancy_threshold
        )
        self._prev_occupancy_map = np.logical_and(
            self._prev_occupancy_map, lidar_map > self._empty_threshold
        )
        self._occupancy_grid = block_reduce(
            self._prev_occupancy_map,
            block_size=self._map_size_pixels // self._reduced_map_size,
            func=np.max,
        )

    def set_state(self, state: np.ndarray) -> None:
        """Sets the state directly (for simulation/testing)."""
        self._state = state.copy().astype(np.float64)

    @property
    def occupancy_grid(self) -> np.ndarray | None:
        """Returns the current occupancy grid for planning."""
        return self._occupancy_grid
