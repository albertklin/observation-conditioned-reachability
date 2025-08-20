import numpy as np
from scipy.interpolate import RegularGridInterpolator

class OnlineEnvironment:

    def __init__(self, X, Y, TH):
        self.coordinate_vectors = [X, Y, TH]
        self.occupancy_grid = np.zeros((len(X), len(Y), len(TH)), dtype=bool)

    def set_occupancy_grid(self, grid):
        self.occupancy_grid = grid
        
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
        if use_cupy:
            print('use_cupy not implemented...')
        interpolator = RegularGridInterpolator
        return interpolator(coordinate_vectors, grid, bounds_error=False, fill_value=None)(states)