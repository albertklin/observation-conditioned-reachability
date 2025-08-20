import numpy as np

from utils.navigation_task import NavigationTask

class PredictiveSampler:
    """A predictive sampler that returns the best control sequence found with predictive sampling."""
    
    def __init__(
            self,
            navigation_task: NavigationTask,
            num_samples: int,
            control_min: "np.ndarray[np.float_]",
            control_max: "np.ndarray[np.float_]",
            noise_scale: "np.ndarray[np.float_]",
            dt: float,
            use_shortest_paths = False,
    ):
        """Initializes a predictive sampler.
        
        Args:
            navigation_task: A navigation task.
            num_samples: The number of samples evaluated.
            control_min: A numpy array with shape [control_dim].
            control_max: A numpy array with shape [control_dim].
            noise_scale: A numpy array with shape [control_dim]. The standard deviation of the distribution of control samples.
            dt: The timestep of the simulation.
        """
        self.navigation_task = navigation_task
        self.num_samples = num_samples
        self.control_min = control_min
        self.control_max = control_max
        self.noise_scale = noise_scale
        self.dt = dt
        self.use_shortest_paths = use_shortest_paths

    def optimal_control_sequence(
            self,
            initial_state: "np.ndarray[np.float_]",
            nominal_control_sequence: "np.ndarray[np.float_]",
            recompute_shortest_paths_cost_grid=False,
    ):
        """Returns the optimal control sequence found via predictive sampling.
        
        Args:
            initial_state: A numpy array with shape [state_dim].
            nominal_control_sequence: A numpy array with shape [control_sequence_length, control_dim].
        """
        candidate_control_sequences = nominal_control_sequence*np.ones((self.num_samples, *nominal_control_sequence.shape))
        for i in range(len(self.noise_scale)):
            candidate_control_sequences[:, :, i] = candidate_control_sequences[:, :, i] + np.random.normal(scale=self.noise_scale[i], size=candidate_control_sequences.shape[:2])
        candidate_control_sequences = np.maximum(candidate_control_sequences, self.control_min)
        candidate_control_sequences = np.minimum(candidate_control_sequences, self.control_max)
        initial_states = initial_state*np.ones((self.num_samples, len(initial_state)))
        costs = self.navigation_task.cost(
            initial_states,
            candidate_control_sequences,
            self.dt,
            use_shortest_paths=self.use_shortest_paths,
            recompute_shortest_paths_cost_grid=recompute_shortest_paths_cost_grid,
        )
        return candidate_control_sequences[np.argmin(costs)]