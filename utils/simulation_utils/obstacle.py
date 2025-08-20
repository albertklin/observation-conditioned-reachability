import abc
import torch
import numpy as np

class Obstacle(metaclass=abc.ABCMeta):
    """An abstract base class for representing an environment obstacle."""

    @abc.abstractmethod
    def distances(
            self,
            positions: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        """Returns signed distances from the positions to the obstacle.
        
        Args:
            positions: A numpy array with shape [batch_size, 2].

        Returns:
            distances: A numpy array with shape [batch_size].
        """

class CircularObstacle(Obstacle):
    """A circular obstacle in an environment."""

    def __init__(
            self,
            center: "list[float]",
            radius: float,
            height: float,
    ):
        """Initializes a circular obstacle.
        
        Args:
            center: The [x, y] position of the obstacle center.
            radius: The radius of the obstacle.
            height: The height of the obstacle.
        """
        self.center = np.array(center)
        self.radius = radius
        self.height = height

    def distances(
            self,
            positions: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        return np.linalg.norm(
            positions - self.center,
            axis=-1
        ) - self.radius
    
    def distances_and_gradients(self, positions):
        positions = torch.tensor(positions, requires_grad=True)
        distances = torch.linalg.norm(positions[..., :2] - torch.tensor(self.center), dim=-1) - self.radius
        distances.sum().backward()
        return distances.detach().numpy(), positions.grad.detach().numpy()

class BoxObstacle(Obstacle):
    """A box obstacle in an environment."""

    def __init__(
            self,
            center: "list[float]",
            angle: float,
            length: float,
            width: float,
            height: float,
    ):
        """Initializes a box obstacle.
        
        Args:
            center: The [x, y] position of the obstacle center.
            angle: The angle of the obstacle w.r.t. the x-axis.
            length: The length of the obstacle (along the ego x-axis).
            width: The width of the obstacle (along the ego y-axis).
            height: The height of the obstacle (along the ego z-axis).
        """
        self.center = np.array(center)
        self.angle = angle
        self.length = length
        self.width = width
        self.height = height

    def distances(
            self,
            positions: "np.ndarray[np.float_]",
    ) -> "np.ndarray[np.float_]":
        # compute positions in ego frame
        positions = positions.copy()
        cth, sth = np.cos(self.angle), np.sin(self.angle)
        rot = np.array([
            [cth, sth],
            [-sth, cth],
        ])
        positions[:, :2] = np.matmul(rot, (positions[:, :2] - self.center)[:, :, np.newaxis]).squeeze(axis=-1)
        # compute signed distances along the ego x-axis to the boundaries that are aligned with the ego y-axis
        dxs = np.maximum(-positions[:, 0]-self.length/2, positions[:, 0]-self.length/2)
        # compute signed distances along the ego y-axis to the boundaries that are aligned with the ego x-axis
        dys = np.maximum(-positions[:, 1]-self.width/2, positions[:, 1]-self.width/2)
        # combine the signed distances: https://stackoverflow.com/questions/30545052/calculate-signed-distance-between-point-and-rectangle
        return np.minimum(0, np.maximum(dxs, dys)) + np.linalg.norm(np.maximum(0, np.stack((dxs, dys), axis=-1)), axis=-1)
    
    def distances_and_gradients(self, positions):
        # compute positions in ego frame
        positions = torch.tensor(positions, requires_grad=True)
        cth, sth = np.cos(self.angle), np.sin(self.angle)
        rot = torch.tensor([
            [cth, sth],
            [-sth, cth],
        ])
        new_px_py = torch.matmul(rot, (positions[:, :2] - torch.tensor(self.center))[:, :, np.newaxis]).squeeze(dim=-1)
        # compute signed distances along the ego x-axis to the boundaries that are aligned with the ego y-axis
        dxs = torch.maximum(-new_px_py[:, 0]-self.length/2, new_px_py[:, 0]-self.length/2)
        # compute signed distances along the ego y-axis to the boundaries that are aligned with the ego x-axis
        dys = torch.maximum(-new_px_py[:, 1]-self.width/2, new_px_py[:, 1]-self.width/2)
        # combine the signed distances: https://stackoverflow.com/questions/30545052/calculate-signed-distance-between-point-and-rectangle
        distances = torch.minimum(torch.tensor(0), torch.maximum(dxs, dys)) + torch.linalg.norm(torch.maximum(torch.tensor(0), torch.stack((dxs, dys), dim=-1)), dim=-1)
        distances.sum().backward()
        return distances.detach().numpy(), positions.grad.detach().numpy()