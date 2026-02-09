#!/usr/bin/env python3
"""Unified launcher for OCR safety filter deployment.

Reads configuration and spawns the appropriate nodes based on the
'nodes' section of the config. Supports both hardware and simulation.

Usage:
    # Simulation (lightweight dynamics)
    python -m hardware.launch --config hardware/configs/simulation.yaml

    # Hardware deployment
    python -m hardware.launch --config hardware/configs/hardware.yaml

The launcher spawns each node in a separate terminal window (requires xterm)
and runs the safety filter in the foreground.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Node module paths
NODE_MODULES = {
    # Lidars
    "rplidar": "hardware.nodes.lidars.rplidar",
    # State estimators
    "slam": "hardware.nodes.state_estimators.slam",
    # Planners
    "mps": "hardware.nodes.planners.mps",
    # Robots (for simulation)
    "simulated": "hardware.nodes.robots.simulated",
    # Visualization
    "visualization": "hardware.visualization",
}


def get_node_command(node_type: str, node_impl: str, config_path: str, extra_args: dict) -> list:
    """Get the command to launch a node."""
    if node_impl not in NODE_MODULES:
        raise ValueError(f"Unknown node implementation: {node_impl}")

    module = NODE_MODULES[node_impl]

    # rplidar doesn't accept --config, only --port and --baudrate
    if node_impl == "rplidar":
        cmd = [sys.executable, "-m", module]
        if "lidar_port" in extra_args:
            cmd.extend(["--port", extra_args["lidar_port"]])
        return cmd

    cmd = [sys.executable, "-m", module, "--config", config_path]

    # Add extra arguments
    if node_type == "planner":
        if "goal_x" in extra_args:
            cmd.extend(["--goal-x", str(extra_args["goal_x"])])
        if "goal_y" in extra_args:
            cmd.extend(["--goal-y", str(extra_args["goal_y"])])

    return cmd


def spawn_in_terminal(name: str, cmd: list) -> subprocess.Popen:
    """Spawn a command in a new terminal window."""
    # Try xterm first, then gnome-terminal, then konsole
    terminal_cmds = [
        ["xterm", "-hold", "-title", name, "-e"] + cmd,
        ["gnome-terminal", "--title", name, "--"] + cmd,
        ["konsole", "--hold", "-e"] + cmd,
    ]

    for term_cmd in terminal_cmds:
        try:
            return subprocess.Popen(term_cmd)
        except FileNotFoundError:
            continue

    # Fallback: run in background without terminal
    print(f"Warning: No terminal emulator found. Running {name} in background.")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(
        description="OCR Safety Filter Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run simulation
    python -m hardware.launch --config hardware/configs/simulation.yaml

    # Run hardware deployment
    python -m hardware.launch --config hardware/configs/hardware.yaml

    # Specify goal position
    python -m hardware.launch --config hardware/configs/simulation.yaml --goal-x 5 --goal-y 2
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hardware/configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--goal-x",
        type=float,
        default=None,
        help="Goal X position (overrides config)",
    )
    parser.add_argument(
        "--goal-y",
        type=float,
        default=None,
        help="Goal Y position (overrides config)",
    )
    parser.add_argument(
        "--lidar-port",
        type=str,
        default=None,
        help="LiDAR serial port (overrides auto-detection)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    # Check config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    nodes_config = config.get("nodes", {})

    # Collect extra args
    extra_args = {}
    if args.goal_x is not None:
        extra_args["goal_x"] = args.goal_x
    if args.goal_y is not None:
        extra_args["goal_y"] = args.goal_y
    if args.lidar_port is not None:
        extra_args["lidar_port"] = args.lidar_port

    # Auto-detect LiDAR port for hardware
    if nodes_config.get("lidar") == "rplidar" and "lidar_port" not in extra_args:
        for port in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]:
            if os.path.exists(port):
                extra_args["lidar_port"] = port
                print(f"Auto-detected LiDAR port: {port}")
                break
        else:
            print("Warning: No LiDAR port detected. Use --lidar-port to specify.")

    print("=" * 50)
    print("OCR Safety Filter Launcher")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Nodes: {nodes_config}")
    print()

    # Track spawned processes
    processes = []

    def cleanup(signum=None, frame=None):
        """Clean up spawned processes."""
        print("\nShutting down...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Determine launch order based on nodes config
        # For simulation: robot simulator first (publishes sensor data)
        # For hardware: lidar -> state_estimator -> planner

        if nodes_config.get("robot") == "simulated":
            # Simulation mode
            print("Mode: Simulation")
            print()

            # 1. Launch simulated robot (publishes LIDAR_SCAN, ROBOT_STATE)
            cmd = get_node_command("robot", "simulated", args.config, extra_args)
            print(f"Starting simulated robot: {' '.join(cmd)}")
            if not args.dry_run:
                proc = spawn_in_terminal("Simulated Robot", cmd)
                processes.append(("Simulated Robot", proc))
                time.sleep(1)

        else:
            # Hardware mode
            print("Mode: Hardware")
            print()

            # 1. Launch LiDAR publisher
            lidar_impl = nodes_config.get("lidar")
            if lidar_impl:
                cmd = get_node_command("lidar", lidar_impl, args.config, extra_args)
                print(f"Starting LiDAR publisher: {' '.join(cmd)}")
                if not args.dry_run:
                    proc = spawn_in_terminal("LiDAR Publisher", cmd)
                    processes.append(("LiDAR Publisher", proc))
                    time.sleep(1)

            # 2. Launch state estimator
            state_impl = nodes_config.get("state_estimator")
            if state_impl:
                cmd = get_node_command("state_estimator", state_impl, args.config, extra_args)
                print(f"Starting state estimator: {' '.join(cmd)}")
                if not args.dry_run:
                    proc = spawn_in_terminal("State Estimator", cmd)
                    processes.append(("State Estimator", proc))
                    time.sleep(1)

        # 3. Launch planner (same for both modes)
        planner_impl = nodes_config.get("planner")
        if planner_impl:
            cmd = get_node_command("planner", planner_impl, args.config, extra_args)
            print(f"Starting planner: {' '.join(cmd)}")
            if not args.dry_run:
                proc = spawn_in_terminal("MPS Planner", cmd)
                processes.append(("MPS Planner", proc))
                time.sleep(1)

        # 4. Launch visualization
        cmd = get_node_command("visualization", "visualization", args.config, extra_args)
        print(f"Starting visualization: {' '.join(cmd)}")
        if not args.dry_run:
            proc = spawn_in_terminal("Visualization", cmd)
            processes.append(("Visualization", proc))
            time.sleep(1)

        # 5. Launch safety filter in foreground
        print()
        print("=" * 50)
        print("Starting safety filter (foreground)...")
        print("Press SPACE/P to unpause, Q to quit")
        print("=" * 50)
        print()

        filter_cmd = [
            sys.executable, "-m", "hardware.filter",
            "--config", args.config,
        ]

        # Start unpaused in simulation mode
        if nodes_config.get("robot") == "simulated":
            filter_cmd.append("--no-pause")

        if args.dry_run:
            print(f"[DRY RUN] {' '.join(filter_cmd)}")
        else:
            # Run filter in foreground
            subprocess.run(filter_cmd)

    finally:
        cleanup()


if __name__ == "__main__":
    main()
