import abc
import numpy as np

class Dynamics(metaclass=abc.ABCMeta):
    """An abstract base class for representing dynamics."""

    def __init__(
            self,
            periodic_dims: "list[int]",
            state_min: "list[float]",
            state_max: "list[float]",
    ):
        """Initializes dynamics.
        
        Args:
            periodic_dims: A list of indices specifying the periodic dimensions of the state space. Can be None.
            state_min: The lower bound of the state space. Dimensions that are not periodic can specify None.
            state_max: The upper bound of the state space. Dimensions that are not periodic can specify None.
        """
        self.periodic_dims = periodic_dims
        self.state_min = state_min
        self.state_max = state_max

    @abc.abstractmethod
    def dynamics(
            self,
            states: "np.ndarray[np.float_]",
            controls: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        """Returns the dynamics f(states, controls).
        
        Args:
            states: A numpy array with shape [batch_size, state_dim].
            controls: A numpy array with shape [batch_size, control_dim].

        Returns:
            dynamics: A numpy array with shape [batch_size, state_dim].
        """

    def wrap_states(
            self,
            states: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        """Returns the states wrapped in the periodic dynamics state space.
        
        Args:
            states: A numpy array with shape [batch_size, state_dim].

        Returns:
            wrapped_states: A numpy array with shape [batch_size, state_dim].
        """
        if self.periodic_dims is None:
            return states
        wrapped_states = states.copy()
        periodic_states = wrapped_states[:, self.periodic_dims]
        periodic_state_min = np.array([self.state_min[i] for i in self.periodic_dims])
        periodic_state_max = np.array([self.state_max[i] for i in self.periodic_dims])
        periodic_states = (periodic_states - periodic_state_min) % (periodic_state_max-periodic_state_min) + periodic_state_min
        wrapped_states[:, self.periodic_dims] = periodic_states
        return wrapped_states
    
    def unwrap_states(
            self,
            states: "np.ndarray[np.float_]"
    ) -> "np.ndarray[np.float_]":
        """Returns the states unwrapped out of the periodic dynamics state space.
        
        Args:
            states: A numpy array with shape [batch_size, state_dim].

        Returns:
            unwrapped_states: A numpy array with shape [batch_size, state_dim].
        """
        if self.periodic_dims is None:
            return states
        unwrapped_states = states.copy()
        periodic_states = unwrapped_states[:, self.periodic_dims]
        periodic_state_min = np.array([self.state_min[i] for i in self.periodic_dims])
        periodic_state_max = np.array([self.state_max[i] for i in self.periodic_dims])
        periodic_states = np.unwrap(periodic_states, axis=0, period=periodic_state_max-periodic_state_min)
        unwrapped_states[:, self.periodic_dims] = periodic_states
        return unwrapped_states
    
    def runge_kutta_step(
            self,
            states: "np.ndarray[np.float_]",
            controls: "np.ndarray[np.float_]",
            timesteps: "np.ndarray[np.float_] | float",
    ) -> "np.ndarray[np.float_]":
        """Returns the 4th-order Runge-Kutta step.
        See "Runge-Kutta Integrator Overview" by Steve Brunton.
        NOTE: Does NOT wrap states.
        
        Args:
            states: A numpy array with shape [batch_size, state_dim].
            controls: A numpy array with shape [batch_size, control_dim].
            timesteps: A numpy array with shape [batch_size] or a single scalar.

        Returns:
            next_states: A numpy array with shape [batch_size, state_dim].
        """
        if isinstance(timesteps, float):
            timesteps = timesteps*np.ones(len(states))
        timesteps = timesteps[:, np.newaxis]
        f1 = self.dynamics(states, controls)
        f2 = self.dynamics(states + f1*timesteps/2, controls)
        f3 = self.dynamics(states + f2*timesteps/2, controls)
        f4 = self.dynamics(states + f3*timesteps, controls)
        return states + (f1 + 2*f2 + 2*f3 + f4)*timesteps/6

class Dubins3D(Dynamics):
    """Dubins3D dynamics implementation.
    
    The dynamics are:
    dx  = v * cos(th) + c1
    dy  = v * sin(th) + c2
    dth = w           + c3

    with controls:
    [v, w]
    """

    def __init__(self):
        super().__init__(
            periodic_dims=[2],
            state_min=[None, None, -np.pi],
            state_max=[None, None, np.pi]
        )
        self.c1, self.c2, self.c3 = 0, 0, 0

    def dynamics(self, states, controls):
        dynamics = np.zeros_like(states)
        dynamics[:, 0] = controls[:, 0] * np.cos(states[:, 2]) + self.c1
        dynamics[:, 1] = controls[:, 0] * np.sin(states[:, 2]) + self.c2
        dynamics[:, 2] = controls[:, 1] + self.c3
        return dynamics
    
    def set_bias(self, c1, c2, c3):
        self.c1, self.c2, self.c3 = c1, c2, c3