# Observation-Conditioned Reachability (OCR)

This repository hosts the code used in [One Filter to Deploy Them All](https://sia-lab-git.github.io/One_Filter_to_Deploy_Them_All/).

## Setup

Install packages in a Python3.8 virtual environment.

```bash
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip

pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade hj-reachability
pip install -e libraries/walk-these-ways
pip install -e libraries/BreezySLAM/python
pip install -r requirements.txt
```

Install IsaacGym:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xf isaac-gym-preview-4 -C libraries
rm isaac-gym-preview-4
pip install -e libraries/isaacgym/python
```

Test your IsaacGym installation:

```bash
cd libraries/isaacgym/python/examples
python joint_monkey.py
```

For systems with both Intel and NVIDIA GPUs, you might need to run:

```bash
sudo prime-select nvidia
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

## Test Your Python Environment

```bash
python scripts/test_python_environment.py
```

## Generate Data for OCR-VN Training

```bash
python scripts/generate_value_network_data.py
```

Generating the training and validation dataset should take ~30 minutes on the latest NVIDIA GPU models. If you encounter `RuntimeError: Unable to load cuSPARSE. Is it installed?` from the Jax library, you may need to first run `unset LD_LIBRARY_PATH`.

If you get a JAX CUDA error on a GPU with <20GB, it could be due to memory issues. Try reducing the `state_grid_shape` on Line 43.

## Train the OCR-VN

```bash
python scripts/train_value_network.py
```

Training the OCR-VN on the latest NVIDIA GPU models should take ~8 hours.

## Download a Trained Checkpoint Model from Hugging Face

```bash
huggingface-cli login  # OR set environment variable HUGGINGFACE_TOKEN
```

Get access to the project here: https://huggingface.co/datasets/albertkuilin/observation-conditioned-reachability

```bash
python scripts/sync_data.py pull results results
```

## Inspect the OCR-VN Prediction In a Validation Environment

```bash
# Predictions across the x-y space using the centroidal LiDAR scan
python scripts/visualize_value_network_single_lidar.py

# Predictions across the x-y space using the local LiDAR scan
python scripts/visualize_value_network_multi_lidar.py
```

## Calibrate the Value Network

The safety filter uses a calibration adjustment to ensure conservative predictions. To compute this for your trained model:

```bash
python scripts/calibrate_value_network.py
```

This uses conformal prediction to compute probabilistic error bounds. The script outputs both the `error_bound` and the corresponding `calibration_adjustment` (its negation) for each epsilon level.

## Generate Simulation Environments

```bash
python scripts/generate_sims.py
```

You can also pull simulation environments with `python scripts/sync_data.py pull data data`.

These correspond to what is described in the paper as "hard" environments, where large dynamical uncertainty is introduced by large variations in the floor friction.

## Run the OCR Safety Filter In Simulation

```bash
python scripts/run_sims.py
```

Analyze the results with:

```bash
python scripts/analyze_sims.py
```

These steps should reproduce the results in Table III of the paper (specifically, PS + WTW OCR). Other results should be reproducible by setting the parameters as specified in the paper.

## Hardware Deployment

The `hardware/` directory provides a multi-process LCM-based architecture for deploying the OCR safety filter on real robots. **Note:** This code provides the core safety filtering logic and reference implementations—you may need to adapt hardware-specific interfaces for your own robot platform, sensors, and compute setup.

### Our Hardware Setup

The hardware experiments in the paper used the following setup:

| Component | Model | Notes |
|-----------|-------|-------|
| Robot | Unitree Go1 EDU | Quadruped robot with low-level SDK access |
| LiDAR | SLAMTEC RPLidar A2 | 360° scan, mounted on robot body |
| Onboard Compute | Intel NUC | Runs safety filter and SLAM |
| State Estimation | BreezySLAM | SLAM-based localization using LiDAR |
| Communication | LCM | Inter-process communication between modules |

The LiDAR was mounted with an offset from the robot's center of mass (configurable in the training options).

### System Architecture

The deployment uses a multi-process architecture with LCM (Lightweight Communications and Marshalling) for inter-process communication:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         External Computer (Intel NUC)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   RPLidar    │    │  SLAM Node   │    │ MPS Planner  │                   │
│  │  Publisher   │───▶│ (BreezySLAM) │───▶│ (sampling-   │                   │
│  └──────────────┘    └──────────────┘    │  based)      │                   │
│         │                   │            └──────────────┘                   │
│         │                   │                   │                            │
│         │                   │  state            │ nominal command            │
│         │                   ▼                   ▼                            │
│         │            ┌─────────────────────────────────────────┐            │
│         └───────────▶│           Safety Filter                 │            │
│              lidar   │  • Receives: state, LiDAR, nominal cmd  │            │
│                      │  • Computes: value, gradient, QP        │            │
│                      │  • Outputs: filtered velocity command   │            │
│                      └─────────────────────────────────────────┘            │
│                                        │                                     │
└────────────────────────────────────────│─────────────────────────────────────┘
                                         │ velocity command (ethernet)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Unitree Go1 (internal Jetson Nano)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                      ┌─────────────────────────────────────────┐            │
│                      │         Walk-These-Ways                 │            │
│                      │    (learned locomotion controller)      │            │
│                      └─────────────────────────────────────────┘            │
│                                        │                                     │
│                                        ▼                                     │
│                               Motor Commands (SDK)                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

For the low-level Walk-These-Ways controller setup, see `libraries/walk-these-ways/README.md` which includes detailed deployment instructions for the Unitree Go1.

### Structure

```
hardware/
├── launch.py              # Unified launcher (spawns all nodes)
├── filter.py              # Safety filter node
├── visualization.py       # Real-time monitoring GUI
├── configs/
│   ├── default.yaml       # Base configuration
│   ├── simulation.yaml    # Simulation mode config
│   └── hardware.yaml      # Hardware deployment config
├── nodes/                 # LCM node implementations
│   ├── lidars/
│   │   └── rplidar.py     # RPLidar sensor driver
│   ├── state_estimators/
│   │   └── slam.py        # BreezySLAM state estimation
│   ├── planners/
│   │   └── mps.py         # Model Predictive Sampling planner
│   └── robots/
│       └── simulated.py   # Simulated robot for testing
├── interfaces/            # Abstract interfaces
│   ├── state_estimator.py # Localization interface
│   ├── lidar.py           # LiDAR sensor interface
│   └── robot.py           # Robot control interface
├── sim/                   # Simulated implementations
│   ├── isaac_gym_sim.py   # IsaacGym simulation
│   ├── sim_state_estimator.py
│   ├── sim_lidar.py
│   └── sim_robot.py
└── lcm_types/             # LCM message definitions
    ├── lidar_scan_t.py    # Raw LiDAR data
    ├── robot_state_t.py   # Pose estimates
    ├── velocity_command_t.py  # Control commands
    ├── occupancy_map_t.py # SLAM occupancy grid
    ├── mps_trajectory_t.py    # Planner trajectory
    └── safety_status_t.py # Filter status for logging
```

### LCM Setup

LCM uses multicast UDP for communication. Before running, ensure multicast routing is configured:

```bash
# Enable multicast on loopback (required for single-machine testing)
sudo ip route add 224.0.0.0/4 dev lo

# Verify LCM works
python -c "import lcm; lc = lcm.LCM(); print('LCM OK')"
```

### Quick Start: Simulation

Run a complete simulation with the unified launcher:

```bash
python -m hardware.launch --config hardware/configs/simulation.yaml
```

This spawns the simulated robot, MPS planner, visualization, and safety filter. The robot navigates toward the goal while the safety filter prevents collisions.

### Quick Start: Hardware

For hardware deployment with RPLidar + BreezySLAM:

```bash
python -m hardware.launch --config hardware/configs/hardware.yaml --goal-x 10 --goal-y 0
```

### LCM Channels

| Channel | Message Type | Publisher | Description |
|---------|--------------|-----------|-------------|
| `LIDAR_SCAN` | `lidar_scan_t` | LiDAR node | Raw 360° scan data |
| `ROBOT_STATE` | `robot_state_t` | SLAM node | Pose estimate (x, y, θ) |
| `OCCUPANCY_MAP` | `occupancy_map_t` | SLAM node | Occupancy grid for planning |
| `NOMINAL_COMMAND` | `velocity_command_t` | Planner | Unfiltered velocity command |
| `SAFE_COMMAND` | `velocity_command_t` | Safety filter | Filtered velocity command |
| `SAFETY_STATUS` | `safety_status_t` | Safety filter | Filter status for logging |
| `MPS_TRAJECTORY` | `mps_trajectory_t` | Planner | Planned trajectory for visualization |

### Keyboard Controls

The safety filter supports keyboard controls (requires `pynput`):

| Key | Action |
|-----|--------|
| `space` / `p` | Pause/unpause |
| `q` | Quit |
| `i` | Immobilize (force zero velocity) |

Hardware mode starts **paused** for safety.

### Visualization

The launcher automatically opens real-time visualization windows showing:

- **Robot State & LiDAR (World Frame)**: Current position, heading, trajectory, and raw LiDAR scan
- **Network Input (Ego Frame)**: Preprocessed LiDAR as seen by the neural network
- **Safety Filter Monitor**: Time-series plots of value function, velocity commands, and disturbance bounds
- **MPS Planner**: Occupancy map and planned trajectory

Visualization requires PyQt5 and pyqtgraph (included in `requirements.txt`).

### Deploying on Your Hardware

To adapt to your hardware:

1. **Implement nodes that publish to the required LCM channels:**
   - `LIDAR_SCAN`: Raw LiDAR data from your sensor (angles in radians [-pi, pi], distances in meters)
   - `ROBOT_STATE`: Pose estimates from your state estimator (x, y in meters relative to start, theta in radians [-pi, pi])
   - `NOMINAL_COMMAND`: Velocity commands from your planner or teleop (v_x in m/s, v_yaw in rad/s)

2. **Implement a node that subscribes to `SAFE_COMMAND`** to send filtered velocities to your robot's locomotion controller

3. **Copy and modify** a config file for your setup (control bounds, sensor parameters, LiDAR offset, etc.)

See `hardware/nodes/` for reference implementations. The `hardware/interfaces/` directory contains abstract base classes, and `hardware/sim/` provides simulated implementations for testing.

### Configuration

The YAML config file controls all parameters:

```yaml
# Node implementations (set to null to skip launching)
nodes:
  lidar: rplidar        # or null
  state_estimator: slam # or null
  planner: mps          # or null
  robot: null           # set to "simulated" for testing

# Model paths
model:
  checkpoint: "results/training/checkpoints/epoch_05000.pth"
  options: "results/training/options.pickle"

# Safety filter parameters
filter:
  threshold: 0.35
  calibration_adjustment: -0.5  # Negated output from calibrate_value_network.py
  slack_coeff: 1000.0

# Control bounds (adjust for your robot)
control:
  v_x_min: 0.0
  v_x_max: 2.0
  v_yaw_min: -2.0
  v_yaw_max: 2.0

# See hardware/configs/default.yaml for full options
```

## Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{11301616,
  author={Lin, Albert and Peng, Shuang and Bansal, Somil},
  journal={IEEE Transactions on Robotics},
  title={One Filter to Deploy Them All: Robust Safety for Quadrupedal Navigation in Unknown Environments},
  year={2026},
  volume={42},
  pages={545-560},
  doi={10.1109/TRO.2025.3644957}
}
```
