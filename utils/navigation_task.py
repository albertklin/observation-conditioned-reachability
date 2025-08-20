import math
import numpy as np

from scipy.ndimage import binary_dilation, minimum_filter

# from utils.hj_reachability_utils.environment import Environment
from utils.dynamics import Dynamics

class NavigationTask:
    """A navigation task that specifies a goal position.
    
    Uses the Environment and Dynamics classes to compute the task cost function.
    """

    def __init__(
            self,
            robot_radius: float,
            goal_position: "np.ndarray[np.float_]",
            # environment: Environment,
            environment,
            dynamics: Dynamics,
            goal_radius: float = 1e-3,
    ):
        """Initializes a navigation task.
        
        Args:
            robot_radius: The radius of the robot that is used for collision checking and goal reaching.
            goal_position: The goal position of the system. It should be a numpy array [x, y].
            environment: The environment. It is assumed to have an occupancy grid whose first two dimensions correspond to x,y and are evenly spaced.
            dynamics: The dynamics.
            goal_radius: The radius of the goal.
        """
        self.robot_radius = robot_radius
        self.goal_position = goal_position
        self.goal_radius = goal_radius
        self.environment = environment
        self.dynamics = dynamics

    def is_at_goal(
            self,
            state: "np.ndarray[np.float_]",
    ) -> bool:
        """Returns True if the robot is within the goal radius and False otherwise.
        
        Args:
            state: A numpy array with shape [state_dim].
        """
        return np.linalg.norm(state[:2]-self.goal_position, axis=-1) <= self.goal_radius + self.robot_radius
    
    def cost(
            self,
            initial_states: "np.ndarray[np.float_]",
            control_sequences: "np.ndarray[np.float_]",
            timesteps: "np.ndarray[np.float_] | float",
            collision_cost: float = 1e9,
            use_shortest_paths = False,
            recompute_shortest_paths_cost_grid = False,
    ) -> "np.ndarray[np.float_]":
        """Returns the costs of the rolled-out trajectories.
        
        Args:
            initial_states: A numpy array with shape [batch_size, state_dim].
            control_sequences: A numpy array with shape [batch_size, ctrl_seq_len, control_dim].
            timesteps: A numpy array with shape [batch_size] or a float.
        
        Returns:
            costs: A numpy array with shape [batch_size].
        """
        
        # roll out trajectories
        state_sequences = np.zeros((initial_states.shape[0], control_sequences.shape[1], initial_states.shape[1]))
        for i in range(control_sequences.shape[1]):
            if i == 0:
                states = initial_states
            else:
                states = state_sequences[:, i-1]
            state_sequences[:, i] = self.dynamics.runge_kutta_step(
                states=states,
                controls=control_sequences[:, i],
                timesteps=timesteps,
            )
        # wrap periodic state to environment state space
        state_sequences = self.dynamics.wrap_states(state_sequences.reshape(-1, initial_states.shape[1])).reshape(initial_states.shape[0], control_sequences.shape[1], initial_states.shape[1])
        
        # compute occupancy_grid
        if hasattr(self.environment, 'query_grid'):
            # pad state to the environment state
            environment_state_dim = len(self.environment.coordinate_vectors)
            state_sequences = np.pad(state_sequences, ((0, 0), (0, 0), (0, environment_state_dim-state_sequences.shape[-1])))
            # inflate occupancy grid by robot radius
            xs, ys = self.environment.coordinate_vectors[0], self.environment.coordinate_vectors[1]
            dx, dy = xs[1]-xs[0], ys[1]-ys[0]
            rx, ry = math.ceil(self.robot_radius/dx), math.ceil(self.robot_radius/dy)
            xx, yy = np.meshgrid(np.arange(-rx, rx+1), np.arange(-ry, ry+1), sparse=True, indexing='ij')
            structure = np.sqrt(np.power(xx*dx, 2) + np.power(yy*dy, 2)) <= self.robot_radius
            if environment_state_dim > 2:
                structure = np.expand_dims(structure, axis=tuple(range(2, environment_state_dim)))
            occupancy_grid = binary_dilation(self.environment.occupancy_grid, structure)
        
        if use_shortest_paths:
            # compute costs with respect to shortest unobstructed paths to the goal_position
            if not hasattr(self.environment, 'query_grid'):
                raise NotImplementedError
            if not hasattr(self, 'cost_grid') or recompute_shortest_paths_cost_grid:
                xy_occupancy_grid = np.any(occupancy_grid == 1, axis=-1)
                xx, yy = np.meshgrid(xs, ys, sparse=True, indexing='ij')
                cost_grid = np.sqrt(np.power(xx-self.goal_position[0], 2) + np.power(yy-self.goal_position[1], 2))
                cost_grid = np.where(cost_grid <= self.goal_radius + self.robot_radius, 0, float('inf'))
                cost_grid[xy_occupancy_grid] = float('inf')
                while True:
                    old_cost_grid = cost_grid
                    cost_grid = np.minimum(cost_grid, minimum_filter(cost_grid, size=3, mode='nearest') + 1)
                    cost_grid[xy_occupancy_grid] = float('inf')
                    if np.all(cost_grid == old_cost_grid):
                        break
                cost_grid[xy_occupancy_grid] = collision_cost
                cost_grid = np.ones(occupancy_grid.shape)*np.expand_dims(cost_grid, axis=-1)
                self.cost_grid = cost_grid
            costs = self.environment.query_grid(
                self.cost_grid,
                self.environment.coordinate_vectors,
                state_sequences.reshape((-1, environment_state_dim))
            ).reshape(initial_states.shape[0], control_sequences.shape[1]).sum(axis=-1)
        else:
            # compute costs with respect to collisions and distance heuristics
            if hasattr(self.environment, 'query_grid'):
                # compute collision costs
                collision_costs = collision_cost*self.environment.query_grid(
                    occupancy_grid,
                    self.environment.coordinate_vectors,
                    state_sequences.reshape((-1, environment_state_dim))
                ).reshape(initial_states.shape[0], control_sequences.shape[1])
            elif hasattr(self.environment, 'occupancies'):
                # compute collision costs
                collision_costs = collision_cost*self.environment.occupancies(self.robot_radius, state_sequences[:, :, :2].reshape(-1, 2)).reshape(initial_states.shape[0], control_sequences.shape[1])
            else:
                raise NotImplementedError
            distance_costs = np.linalg.norm(state_sequences[:, :, :2] - self.goal_position, axis=-1)
            costs = np.sum(collision_costs + distance_costs, axis=-1)
        return costs