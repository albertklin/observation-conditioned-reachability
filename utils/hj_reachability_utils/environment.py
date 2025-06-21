import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt

from hj_reachability import dynamics
from scipy.interpolate import RegularGridInterpolator

# import cupy
# from cupyx.scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.hj_reachability_utils.obstacle import Obstacle

class Environment:
    """A base class for representing an environment with obstacles.
    
    At initialization, it computes its own state grid,
    distance grid, occupancy grid, value grid, and grad grid.
    
    Accessible cache includes:
    self.coordinate_vectors: list of length state_dim of the grid coordinates for each dim
    self.state_grid: np.ndarray[np.float_] of shape [*state_grid_shape, state_dim]
    self.distance_grid: np.ndarray[np.float_] of shape state_grid_shape
    self.occupancy_grid: np.ndarray[np.float_] of shape state_grid_shape
    self.value_grid: np.ndarray[np.float_] of shape state_grid_shape
    self.grad_grid: np.ndarray[np.float_] of shape [*state_grid_shape, state_dim]
    """

    def __init__(
        self,
        state_min: "list[float]",
        state_max: "list[float]",
        state_grid_shape: "list[int]",
        state_periodic_dims: "int | list[int] | None",
        obstacles: "list[Obstacle]",
        dynamics: dynamics.Dynamics,
        time_horizon: float,
        batch_grid_shape: "list[int]" = None,
        progress_bar: bool = True,
    ):
        """Initializes an environment with obstacles.

        It computes and caches its own state grid,
        distance grid, occupancy grid, value grid, and grad grid.
        
        Args:
            state_min: The lower bounds of the state space. It should be a list of length state_dim.
            state_max: The upper bounds of the state space. It should be a list of length state_dim.
            state_grid_shape: The shape of the state grid. It should be a list of length state_dim.
            state_periodic_dims: The periodic dimensions of the state space. It can be a single integer, a list of integers, or None.
            obstacles: The obstacles in the environment.
            dynamics: The dynamics of the environment.
            time_horizon: The time horizon of the reachability computation.
            batch_grid_shape: The shape of the batches that the reachability computation is done over, to overcome memory/runtime issues.
                It should be a list of length state_dim and evenly divide into state_grid_shape.
                Be careful to batch only the non-dynamical (virtual) state variables.
            progress_bar: Whether to display the progress_bar for the reachability computation.
        """
        self.state_min = state_min
        self.state_max = state_max
        self.state_grid_shape = state_grid_shape
        self.state_periodic_dims = state_periodic_dims
        self.obstacles = obstacles
        self.dynamics = dynamics
        self.time_horizon = time_horizon
        self.batch_grid_shape = batch_grid_shape

        if batch_grid_shape is not None:
            assert not np.any(np.array(state_grid_shape) % np.array(batch_grid_shape))
        # compute state_grid
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj.sets.Box(np.array(state_min), np.array(state_max)),
            shape=state_grid_shape,
            periodic_dims=state_periodic_dims,
        )
        coordinate_vectors = grid.coordinate_vectors
        state_grid = np.asarray(grid.states)
        # compute distance grid
        distance_grid = obstacles[0].distance_grid(state_grid)
        for obstacle in obstacles:
            distance_grid = np.minimum(obstacle.distance_grid(state_grid), distance_grid)
        # compute occupancy grid
        occupancy_grid = (distance_grid <= 0).astype(np.float64)
        # compute value grid
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        )
        value_grid = hj.step(
            solver_settings,
            dynamics,
            grid,
            time_horizon,
            distance_grid,
            0,
            progress_bar=progress_bar,
        )
        grad_grid = grid.grad_values(value_grid, solver_settings.upwind_scheme)

        # cache all grids
        self.coordinate_vectors = coordinate_vectors
        self.state_grid = state_grid
        self.distance_grid = distance_grid
        self.occupancy_grid = occupancy_grid
        self.value_grid = value_grid
        self.grad_grid = grad_grid

    def visualize_grid(
        self,
        grid: "np.ndarray[np.float_]",
        coordinate_vectors: "list[np.ndarray[np.float_]]",
        axis_dims: "list[int]",
        state_slice_indices: "list[int]",
        save_path: str,
        title: str,
        axis_labels: "list[str]",
        legend_limit: float = None,
        aspect='equal',
        contour_target_set=False,
    ):
        """Saves a visualization of the grid to save_path.
        
        Args:
            grid: The grid to visualize. Should have the same shape as self.distance_grid.
            coordinate_vectors: A list of the coordinate vectors of the grid.
                The number of coordinate vectors should be the same as the number of dimensions of the grid.
                Each coordinate vector should be of the same length as the corresponding dimension of the grid. 
            axis_dims: The dimensions of the grid to visualize on the plot axes. Should be a list of two indices specifying the x and y axes, respectively.
            state_slice_indices: The state slice to visualize, specified by indices.
                Should be a list of indices with length equal to the number of dimensions of the grid.
                Can specify None for the axis_dims.
            save_path: The path to save the plot to.
            title: The title of the plot.
            axis_labels: The labels to use for the plot axes. Should be a list of two strings specifying the x and y axes, respectively.
            legend_limit: The legend colorbar will span -legend_limit to legend_limit. If None, legend_limit is computed from the grid values.
        """
        slice_dims = [i for i in range(len(grid.shape)) if i not in axis_dims]
        grid = grid.transpose((*axis_dims, *slice_dims))[(..., *np.array(state_slice_indices)[slice_dims])]
        if legend_limit is None:
            legend_limit = np.max(np.abs(grid))
        plt.figure()
        plt.pcolormesh(coordinate_vectors[axis_dims[0]],
                       coordinate_vectors[axis_dims[1]],
                       grid.T,
                       cmap='RdBu',
                       vmin=-legend_limit,
                       vmax=legend_limit)
        plt.colorbar()
        plt.contour(coordinate_vectors[axis_dims[0]],
                    coordinate_vectors[axis_dims[1]],
                    grid.T,
                    levels=0,
                    colors='black',
                    linewidths=0.4)
        if contour_target_set:
            plt.contour(coordinate_vectors[axis_dims[0]],
                        coordinate_vectors[axis_dims[1]],
                        self.distance_grid[:, :, 0].T,
                        levels=0,
                        colors='red',
                        linewidths=0.4,
                        linestyles='--')
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.title(title)
        plt.gca().set_aspect(aspect)
        plt.savefig(save_path, bbox_inches='tight', dpi=800)
        plt.close()

    def query_grid(
            self,
            grid: "np.ndarray[np.float_]",
            coordinate_vectors: "list[np.ndarray[np.float_]]",
            states: "np.ndarray[np.float_]",
            use_cupy: bool = False,
    ):
        """Returns the interpolated values of the grid at the specified states.
        
        Args:
            grid: The grid to query. Should have the same shape as self.distance_grid.
            coordinate_vectors: A list of the coordinate vectors of the grid.
                The number of coordinate vectors should be the same as the number of dimensions of the grid.
                Each coordinate vector should be of the same length as the corresponding dimension of the grid.
            states: A numpy array with shape [batch_size, state_dim].
            use_cupy: Whether to use cupy for GPU acceleration. If True, then cupy replaces numpy everywhere.

        Returns:
            values: A numpy array with shape [batch_size].
        """
        # interpolator = CupyRegularGridInterpolator if use_cupy else RegularGridInterpolator
        interpolator = NotImplementedError if use_cupy else RegularGridInterpolator
        return interpolator(coordinate_vectors, grid, bounds_error=False, fill_value=None)(states)
    
    # adapted from: https://github.com/LeCAR-Lab/ABS/blob/main/training/legged_gym/legged_gym/utils/math.py
    def read_lidar(
            self,
            positions: "np.ndarray[np.float_]",
            thetas: "np.ndarray[np.float_]",
            min_distance: float = 0.2,
            max_distance: float = 10,
            range_accuracy: float = 0.99,
            angle_resolution: float = 0.01,
            use_cupy: bool = False,
    ):
        """Returns the simulated lidar readings.

        Args:
            positions: A numpy array with shape [batch_size, 2].
            thetas: A numpy array with shape [batch_size, num_rays].
            min_distance: The minimum distance reading.
            max_distance: The maximum distance reading.
            range_accuracy: The range accuracy of the readings. E.g., a range_accuracy of 0.99 means that the error is <0.01*distance.
                Uniform noise is assumed.
            angle_resolution: The angle resolution of the readings. E.g., an angle resolution of 0.01 rad means the actual reading angle differs by at most 0.01 rad.
                Uniform noise is assumed.
            use_cupy: Whether to use cupy for GPU acceleration. If True, then cupy replaces numpy everywhere.

        Returns:
            readings: A numpy array with shape [batch_size, num_rays]
        """
        # lib = cupy if use_cupy else np
        lib = NotImplementedError if use_cupy else np
        positions = lib.asarray(positions) # [batch_size, 2]
        thetas = lib.asarray(thetas) # [batch_size, num_rays]
        thetas = thetas + angle_resolution*lib.random.uniform(-1, 1, thetas.shape) # [batch_size, num_rays]
        stheta = lib.sin(thetas) # [batch_size, num_rays]
        ctheta = lib.cos(thetas) # [batch_size, num_rays]
        centers = lib.asarray([obstacle.center for obstacle in self.obstacles])[:, lib.newaxis] # [num_obstacles, 1, 2]
        radii = lib.asarray([obstacle.radius for obstacle in self.obstacles])[:, lib.newaxis, lib.newaxis] # [num_obstacles, 1, 1]

        x_positions, y_positions = positions[:, 0:1], positions[:, 1:2] # [batch_size, 1]
        x_centers, y_centers = centers[:, :, 0:1], centers[:, :, 1:2] # [num_obstacles, 1, 1]

        d_c2line = lib.abs(stheta*x_centers - ctheta*y_centers - stheta*x_positions + ctheta*y_positions) # [num_obstacles, batch_size, num_rays]
        d_c2line_square = lib.square(d_c2line) # [num_obstacles, batch_size, num_rays]
        d_c0_square = lib.square(x_centers - x_positions) + lib.square(y_centers - y_positions) # [num_obstacles, batch_size, 1]
        d_0p = lib.sqrt(d_c0_square - d_c2line_square) # [num_obstacles, batch_size, num_rays]
        semi_arc = lib.sqrt(lib.square(radii) - d_c2line_square) # [num_obstacles, batch_size, num_rays]
        raydist = lib.nan_to_num(d_0p - semi_arc, nan=max_distance) # [num_obstacles, batch_size, num_rays]
        check_dir = ctheta*(x_centers-x_positions) + stheta*(y_centers-y_positions) # [num_obstacles, batch_size, num_rays]
        raydist = (check_dir > 0)*raydist + (check_dir <= 0)*max_distance # [num_obstacles, batch_size, num_rays]
        raydist = lib.min(raydist, axis=0) # [batch_size, num_rays]
        error = (1-range_accuracy)*raydist # [batch_size, num_rays]
        raydist = (raydist + error*lib.random.uniform(-1, 1, raydist.shape)).clip(min=min_distance, max=max_distance) # [batch_size, num_rays]
        return raydist
    
    def wrap_states(
            self,
            states: "np.ndarray[np.float_]",
            use_cupy: bool = False,
    ) -> "np.ndarray[np.float_]":
        """Returns the states wrapped in the periodic state space.
        
        Args:
            states: A numpy array with shape [batch_size, state_dim].
            use_cupy: Whether to use cupy for GPU acceleration. If True, then cupy replaces numpy everywhere.

        Returns:
            wrapped_states: A numpy array with shape [batch_size, state_dim].
        """
        if self.state_periodic_dims is None:
            return states
        elif isinstance(self.state_periodic_dims, int):
            periodic_dims = [self.state_periodic_dims]
        elif isinstance(self.state_periodic_dims, list):
            periodic_dims = self.state_periodic_dims
        else:
            raise NotImplementedError
        # lib = cupy if use_cupy else np
        lib = NotImplementedError if use_cupy else np
        wrapped_states = states.copy()
        periodic_states = wrapped_states[:, periodic_dims]
        periodic_state_min = lib.array([self.state_min[i] for i in periodic_dims])
        periodic_state_max = lib.array([self.state_max[i] for i in periodic_dims])
        periodic_states = (periodic_states - periodic_state_min) % (periodic_state_max-periodic_state_min) + periodic_state_min
        wrapped_states[:, periodic_dims] = periodic_states
        return wrapped_states
    
    def unwrap_states(
            self,
            states: "np.ndarray[np.float_]",
            use_cupy: bool = False,
    ) -> "np.ndarray[np.float_]":
        """Returns the states unwrapped out of the periodic state space.
        
        Args:
            states: A numpy array with shape [batch_size, state_dim].
            use_cupy: Whether to use cupy for GPU acceleration. If True, then cupy replaces numpy everywhere.

        Returns:
            unwrapped_states: A numpy array with shape [batch_size, state_dim].
        """
        if self.state_periodic_dims is None:
            return states
        elif isinstance(self.state_periodic_dims, int):
            periodic_dims = [self.state_periodic_dims]
        elif isinstance(self.state_periodic_dims, list):
            periodic_dims = self.state_periodic_dims
        else:
            raise NotImplementedError
        # lib = cupy if use_cupy else np
        lib = NotImplementedError if use_cupy else np
        unwrapped_states = states.copy()
        periodic_states = unwrapped_states[:, periodic_dims]
        periodic_state_min = lib.array([self.state_min[i] for i in periodic_dims])
        periodic_state_max = lib.array([self.state_max[i] for i in periodic_dims])
        periodic_states = lib.unwrap(periodic_states, axis=0, period=periodic_state_max-periodic_state_min)
        unwrapped_states[:, periodic_dims] = periodic_states
        return unwrapped_states
    