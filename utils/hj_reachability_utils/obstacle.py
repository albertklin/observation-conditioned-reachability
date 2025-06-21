import abc
import numpy as np

class Obstacle(metaclass=abc.ABCMeta):
    """An abstract base class for representing an environment obstacle.
    
    Given a state grid, it can compute its own
    distance grid and occupancy grid.
    """

    @abc.abstractmethod
    def distance_grid(
        self,
        state_grid: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        """Returns its distance grid.
        
        Args:
            state_grid: The state grid to compute distances for.
                Its shape should be [..., state_dim].

        Returns:
            The distance grid computed from the state grid.
            Its shape should be [...].
        """

    def occupancy_grid(
        self,
        state_grid: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        """Returns its occupancy grid.
        
        Args:
            state_grid: The state grid to compute the occupancies for.
                Its shape should be [..., state_dim].

        Returns:
            The occupancy grid computed from the state grid.
            Its shape should be [...].
        """
        return (self.distance_grid(state_grid) <= 0).astype(np.float_)

class SphericalObstacle(Obstacle):
    """A spherical obstacle in an environment.
    
    Inherits from the abstract Obstacle class.
    
    Given a state grid, it can compute its own
    distance grid and occupancy grid.
    """

    def __init__(
        self,
        center: "list[float]",
        dims: "list[int]",
        radius: float,
    ):
        """Initializes a spherical obstacle.
        
        Args:
            center: The center state of the obstacle.
                It should be a list of length state_dim (or at least the largest dim in the dims argument).
            dims: The state dimensions that the obstacle resides in. Each specified dim must be smaller than the length of the center argument.
            radius: The radius of the obstacle.
        """
        self.center = np.array(center)
        self.dims = np.array(dims)
        self.radius = radius

    def distance_grid(
        self,
        state_grid: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        return np.linalg.norm(
            state_grid[..., self.dims] - self.center[self.dims],
            axis=-1
        ) - self.radius
    
class CylindricalObstacle2D(SphericalObstacle):
    """A 2D cylindrical obstacle in an environment.
    
    Inherits from SphericalObstacle.
    
    Given a state grid, it can compute its own
    distance grid and occupancy grid.
    It also has a height for visualization purposes only.
    """

    def __init__(
        self,
        center: "list[float]",
        radius: float,
        height: float,
    ):
        """Initializes a 2D cylindrical obstacle in the Dubins4D environment.
        
        Args:
            center: The [x, y] position of the obstacle.
            radius: The radius of the obstacle.
            height: The height of the obstacle (for visualization purposes only).
        """
        super().__init__(center=center, dims=[0, 1], radius=radius)
        self.height = height