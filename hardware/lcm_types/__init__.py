"""LCM message types for the OCR safety filter.

These message definitions enable multi-process communication between
hardware components (LiDAR, SLAM, safety filter, robot controller).

To regenerate Python classes from .lcm files:
    lcm-gen -p *.lcm

See https://lcm-proj.github.io/ for LCM documentation.
"""

from hardware.lcm_types.lidar_scan_t import lidar_scan
from hardware.lcm_types.robot_state_t import robot_state
from hardware.lcm_types.velocity_command_t import velocity_command
from hardware.lcm_types.safety_status_t import safety_status
from hardware.lcm_types.occupancy_map_t import occupancy_map
from hardware.lcm_types.mps_trajectory_t import mps_trajectory

__all__ = [
    "lidar_scan",
    "robot_state",
    "velocity_command",
    "safety_status",
    "occupancy_map",
    "mps_trajectory",
]
