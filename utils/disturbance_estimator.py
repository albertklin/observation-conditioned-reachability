import numpy as np

from utils.dynamics import Dynamics

class DisturbanceEstimator:
    """Estimates the disturbance from an observation history."""

    def __init__(
            self,
            initial_state: "np.ndarray[np.float_]",
            prediction_steps: int,
            window_steps: int,
            coverage: float,
            std_width: float,
            dynamics: Dynamics,
    ):
        """Initializes a disturbance estimator.
        
        Args:
            initial_state: A numpy array with shape [state_dim].
            prediction_steps: The number of steps that a state should be propagated before using it for disturbance estimation.
            window_steps: The number of steps into the past that should be used for disturbance estimation.
            coverage: The proportion of disturbances to use for disturbance estimation.
            std_width: The width in standard deviation units that should be used for disturbance estimation.
            dynamics: The dynamics of the system.
        """
        self.prediction_steps = prediction_steps
        self.window_steps = window_steps
        self.coverage = coverage
        self.std_width = std_width
        self.dynamics = dynamics

        self.states = [initial_state]
        self.controls = []
        self.dts = []

    def store_observation(
            self,
            control: "np.ndarray[np.float_]",
            dt: float,
            next_state: "np.ndarray[np.float_]",
    ):
        """Stores an observation.
        
        Args:
            control: A numpy array with shape [control_dim].
            dt: The timestep to next_state
            next_state: A numpy array with shape [state_dim].
        """
        self.controls.append(control)
        self.dts.append(dt)
        self.states.append(next_state)
        
        if len(self.controls) > self.window_steps + self.prediction_steps - 1:
            self.controls.pop(0)
            self.dts.pop(0)
            self.states.pop(0)
        
    def get_latest_disturbance(self):
        assert len(self.controls) >= self.prediction_steps

        # unwrap states | NOTE: VERY IMPORTANT FOR CORRECT DISTURBANCE ESTIMATION ON PERIODIC STATES
        states = self.dynamics.unwrap_states(np.array(self.states))
        controls = np.array(self.controls)
        dts = np.array(self.dts)

        p = self.prediction_steps
        n = len(states)
        pred_state = states[n-1-p]
        cum_dt = 0
        for i in range(p):
            pred_state = self.dynamics.runge_kutta_step(pred_state[np.newaxis], controls[n-1-p+i][np.newaxis], dts[n-1-p+i][np.newaxis])[0]
            cum_dt += dts[n-1-p+i]
        disturbance = (states[n-1] - pred_state) / cum_dt
        return disturbance
    
    def estimate_disturbance_bounds(self) -> "np.ndarray[np.float_]":
        """Returns the estimated upper and lower bounds of disturbance as a numpy array with shape [2, state_dim]."""
        assert len(self.controls) >= self.prediction_steps

        # unwrap states | NOTE: VERY IMPORTANT FOR CORRECT DISTURBANCE ESTIMATION ON PERIODIC STATES
        states = self.dynamics.unwrap_states(np.array(self.states))
        controls = np.array(self.controls)
        dts = np.array(self.dts)

        p = self.prediction_steps
        n = len(states)
        m = min(self.window_steps, n-p) # window size
        i = n-m-p # start index of the initial window

        # compute window of state predictions
        pred_states_window = states[i:i+m]
        for j in range(p):
            pred_states_window = self.dynamics.runge_kutta_step(pred_states_window, controls[i+j:i+j+m], dts[i+j:i+j+m])

        # compute window of cumulative dts for normalization
        cum_dts_window = np.zeros(m)
        for j in range(p):
            cum_dts_window = cum_dts_window + dts[i+j:i+j+m]

        # compute disturbance bounds
        disturbances = (states[i+p:i+p+m] - pred_states_window) / cum_dts_window[:, np.newaxis]
        disturbance_bounds = np.zeros((2, states.shape[-1]))
        for j in range(states.shape[-1]):
            d = np.sort(disturbances[:, j])[int(len(disturbances)*(1-self.coverage)/2) : len(disturbances) - int(len(disturbances)*(1-self.coverage)/2)]
            mean, std  = np.mean(d, axis=0), np.std(d, axis=0)
            disturbance_bounds[:, j] = mean - self.std_width*std, mean + self.std_width*std
        return disturbance_bounds
    
class SingleStepDisturbanceEstimator:
    """Estimates the disturbance from an observation history of single-step transitions.
    Transitions are specified in full for each store_observation (there is no assumed continuity between observations).
    """

    def __init__(
            self,
            window_steps: int,
            coverage: float,
            std_width: float,
            dynamics: Dynamics,
    ):
        """Initializes a disturbance estimator.
        
        Args:
            window_steps: The number of steps into the past that should be used for disturbance estimation.
            coverage: The proportion of disturbances to use for disturbance estimation.
            std_width: The width in standard deviation units that should be used for disturbance estimation.
            dynamics: The dynamics of the system.
        """
        self.window_steps = window_steps
        self.coverage = coverage
        self.std_width = std_width
        self.dynamics = dynamics

        self.states = []
        self.controls = []
        self.dts = []
        self.next_states = []

    def store_observation(
            self,
            state: "np.ndarray[np.float_]",
            control: "np.ndarray[np.float_]",
            dt: float,
            next_state: "np.ndarray[np.float_]",
    ):
        """Stores an observation.
        
        Args:
            state: A numpy array with shape [state_dim].
            control: A numpy array with shape [control_dim].
            dt: The timestep to next_state
            next_state: A numpy array with shape [state_dim].
        """
        self.states.append(state)
        self.controls.append(control)
        self.dts.append(dt)
        self.next_states.append(next_state)
        if len(self.states) > self.window_steps:
            self.states.pop(0)
            self.controls.pop(0)
            self.dts.pop(0)
            self.next_states.pop(0)
        
    def get_latest_disturbance(self):
        state, control, dt, next_state = self.states[-1], self.controls[-1], self.dts[-1], self.next_states[-1]
        state, next_state = self.dynamics.unwrap_states(np.stack((state, next_state), axis=0))
        pred_state = self.dynamics.runge_kutta_step(state[np.newaxis], control[np.newaxis], dt).squeeze(0)
        return (next_state - pred_state) / dt
    
    def estimate_disturbance_bounds(self) -> "np.ndarray[np.float_]":
        """Returns the estimated upper and lower bounds of disturbance as a numpy array with shape [2, state_dim]."""
        # compute disturbance bounds
        disturbances = []
        for i in range(len(self.states)):
            state, control, dt, next_state = self.states[i], self.controls[i], self.dts[i], self.next_states[i]
            state, next_state = self.dynamics.unwrap_states(np.stack((state, next_state), axis=0))
            pred_state = self.dynamics.runge_kutta_step(state[np.newaxis], control[np.newaxis], dt).squeeze(0)
            disturbances.append((next_state - pred_state) / dt)
        disturbances = np.stack(disturbances, axis=0)
        disturbance_bounds = np.zeros((2, disturbances[0].shape[-1]))
        for j in range(disturbances[0].shape[-1]):
            d = np.sort(disturbances[:, j])[int(len(disturbances)*(1-self.coverage)/2) : len(disturbances) - int(len(disturbances)*(1-self.coverage)/2)]
            mean, std  = np.mean(d, axis=0), np.std(d, axis=0)
            disturbance_bounds[:, j] = mean - self.std_width*std, mean + self.std_width*std
        return disturbance_bounds