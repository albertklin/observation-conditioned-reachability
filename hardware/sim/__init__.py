"""Simulated implementations for testing without hardware.

This module provides two simulation modes:

1. Lightweight simulation (SimStateEstimator, SimLidar, SimRobot):
   - Uses simple Dubins dynamics
   - No external dependencies beyond numpy/scipy
   - Good for quick testing

2. Realistic simulation (IsaacGymSimulator):
   - Uses IsaacGym with Walk-These-Ways quadruped controller
   - Full rigid body dynamics with actuator modeling
   - Requires IsaacGym installation
"""

from hardware.sim.sim_state_estimator import SimStateEstimator
from hardware.sim.sim_lidar import SimLidar, IsaacGymLidar
from hardware.sim.sim_robot import SimRobot

__all__ = ["SimStateEstimator", "SimLidar", "IsaacGymLidar", "SimRobot"]

# IsaacGymSimulator is imported separately to avoid isaacgym dependency
# when using lightweight simulation:
#   from hardware.sim.isaac_gym_sim import IsaacGymSimulator
