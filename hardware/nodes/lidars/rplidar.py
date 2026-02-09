#!/usr/bin/env python3
"""RPLidar publisher node.

Reads from an RPLidar sensor and publishes raw scans via LCM.

LCM Channels:
    Publishes: LIDAR_SCAN

Usage:
    python -m hardware.nodes.lidars.rplidar --port /dev/ttyUSB0
"""

import argparse
import time

import numpy as np

try:
    import lcm
except ImportError:
    raise ImportError(
        "LCM not installed. Install with: pip install lcm\n"
        "See https://lcm-proj.github.io/ for more information."
    )

try:
    from rplidar import RPLidar
except ImportError:
    raise ImportError(
        "rplidar not installed. Install with: pip install rplidar-roboticia"
    )

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from hardware.lcm_types import lidar_scan


class RPLidarNode:
    """RPLidar publisher node."""

    def __init__(self, port: str, baudrate: int = 256000):
        self.port = port
        self.baudrate = baudrate

        # Initialize LiDAR
        print(f"Connecting to RPLidar on {port} at {baudrate} baud...")
        self.lidar = RPLidar(port, baudrate=baudrate)

        # Get device info
        info = self.lidar.get_info()
        print(f"LiDAR info: {info}")

        health = self.lidar.get_health()
        print(f"LiDAR health: {health}")

        # LCM
        self.lc = lcm.LCM()

        print("RPLidar node initialized.")

    def run(self):
        """Main loop - iterate over scans and publish."""
        print("RPLidar node started. Publishing to: LIDAR_SCAN")

        try:
            for scan in self.lidar.iter_scans():
                timestamp = int(time.time() * 1e6)

                # Create message
                msg = lidar_scan()
                msg.timestamp = timestamp
                msg.count = min(len(scan), 1400)

                for i, (quality, angle, distance) in enumerate(scan):
                    if i >= 1400:
                        break
                    # Convert angle from degrees to radians, wrap to [-pi, pi]
                    # RPLidar reports angle in degrees, clockwise from front
                    angle_rad = -np.radians(angle)  # Negate for counter-clockwise
                    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
                    msg.angle[i] = angle_rad
                    msg.range[i] = distance / 1000.0  # Convert mm to meters
                    msg.quality[i] = int(quality)

                self.lc.publish("LIDAR_SCAN", msg.encode())

        except KeyboardInterrupt:
            print("\nRPLidar node stopped.")

        finally:
            self.lidar.stop()
            self.lidar.disconnect()
            print("LiDAR disconnected.")

    def stop(self):
        """Stop the LiDAR."""
        self.lidar.stop()
        self.lidar.disconnect()


def main():
    parser = argparse.ArgumentParser(description="RPLidar Publisher Node")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port for RPLidar (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=256000,
        help="Baudrate for RPLidar (default: 256000)",
    )
    args = parser.parse_args()

    node = RPLidarNode(args.port, args.baudrate)
    node.run()


if __name__ == "__main__":
    main()
