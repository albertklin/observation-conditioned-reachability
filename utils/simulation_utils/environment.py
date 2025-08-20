import numpy as np

from utils.simulation_utils.obstacle import Obstacle, CircularObstacle, BoxObstacle

class Environment:
    """A base class for representing an environment with obstacles.
    """

    def __init__(
        self,
        obstacles: "list[Obstacle]",
        payload: float = 0,
        friction: float = 1,
    ):
        """Initializes an environment with obstacles.

        Args:
            obstacles: The obstacles in the environment.
            payload: The payload in kg.
            friction: The coefficient of static friction and dynamic friction.
        """
        self.obstacles = obstacles
        self.payload = payload
        self.friction = friction
        
    def distances(
            self,
            positions,
    ):
        """Computes the minimum signed distances to the obstacles.
        
        Args:
            positions: The positions at which to compute the minimum distances.
                Should be a numpy array with shape [batch_size, 2].
        
        Returns:
            distances: A numpy array with shape [batch_size].
        """
        distances = np.full(len(positions), float('inf'))
        for obstacle in self.obstacles:
            distances = np.minimum(obstacle.distances(positions), distances)
        return distances
    
    def distances_and_gradients(self, positions):
        distances = np.full(len(positions), float('inf'))
        gradients = np.full(positions.shape, np.nan)
        for obstacle in self.obstacles:
            obstacle_distances, obstacle_gradients = obstacle.distances_and_gradients(positions)
            replace = obstacle_distances < distances
            distances[replace] = obstacle_distances[replace]
            gradients[replace] = obstacle_gradients[replace]
        return distances, gradients

    def occupancies(
            self,
            radius,
            positions,
    ):
        """Computes the occupancies of the positions.
        
        Args:
            radius: The radius to inflate occupied regions by.
            positions: A numpy array with shape [batch_size, 2].

        Returns:
            occupancies: A numpy array with shape [batch_size].
        """
        return (self.distances(positions) <= radius).astype(np.float_)
    
    # adapted from: https://github.com/LeCAR-Lab/ABS/blob/main/training/legged_gym/legged_gym/utils/math.py
    def read_lidar(
            self,
            positions: "np.ndarray[np.float_]",
            thetas: "np.ndarray[np.float_]",
            min_distance: float = 0.2,
            max_distance: float = 10,
            range_accuracy: float = 0.99,
            angle_resolution: float = 0.01,
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

        Returns:
            readings: A numpy array with shape [batch_size, num_rays]
        """
        positions = np.asarray(positions) # [batch_size, 2]
        thetas = np.asarray(thetas) # [batch_size, num_rays]
        thetas = thetas + angle_resolution*np.random.uniform(-1, 1, thetas.shape) # [batch_size, num_rays]
        stheta = np.sin(thetas) # [batch_size, num_rays]
        ctheta = np.cos(thetas) # [batch_size, num_rays]

        # readings for different obstacle types are computed separately
        circular_obstacles = [obs for obs in self.obstacles if isinstance(obs, CircularObstacle)]
        box_obstacles = [obs for obs in self.obstacles if isinstance(obs, BoxObstacle)]

        # compute readings for circular obstacles
        if len(circular_obstacles) > 0:
            centers = np.asarray([obstacle.center for obstacle in circular_obstacles])[:, np.newaxis] # [num_obstacles, 1, 2]
            radii = np.asarray([obstacle.radius for obstacle in circular_obstacles])[:, np.newaxis, np.newaxis] # [num_obstacles, 1, 1]

            x_positions, y_positions = positions[:, 0:1], positions[:, 1:2] # [batch_size, 1]
            x_centers, y_centers = centers[:, :, 0:1], centers[:, :, 1:2] # [num_obstacles, 1, 1]

            d_c2line = np.abs(stheta*x_centers - ctheta*y_centers - stheta*x_positions + ctheta*y_positions) # [num_obstacles, batch_size, num_rays]
            d_c2line_square = np.square(d_c2line) # [num_obstacles, batch_size, num_rays]
            d_c0_square = np.square(x_centers - x_positions) + np.square(y_centers - y_positions) # [num_obstacles, batch_size, 1]
            d_0p = np.sqrt(d_c0_square - d_c2line_square) # [num_obstacles, batch_size, num_rays]
            semi_arc = np.sqrt(np.square(radii) - d_c2line_square) # [num_obstacles, batch_size, num_rays]
            circ_raydist = np.nan_to_num(d_0p - semi_arc, nan=max_distance) # [num_obstacles, batch_size, num_rays]
            check_dir = ctheta*(x_centers-x_positions) + stheta*(y_centers-y_positions) # [num_obstacles, batch_size, num_rays]
            circ_raydist = (check_dir > 0)*circ_raydist + (check_dir <= 0)*max_distance # [num_obstacles, batch_size, num_rays]
            circ_raydist = np.min(circ_raydist, axis=0) # [batch_size, num_rays]
        else:
            circ_raydist = np.full_like(thetas, float('inf'))

        # compute readings for box obstacles
        if len(box_obstacles) > 0:
            box_centers = np.asarray([obstacle.center for obstacle in box_obstacles]) # [num_obstacles, 2]
            box_angles = np.asarray([obstacle.angle for obstacle in box_obstacles]) # [num_obstacles]
            box_lws = np.asarray([[obstacle.length, obstacle.width] for obstacle in box_obstacles]) # [num_obstacles, 2]
            
            # find boundary line segment points, where each segment is represented as [x1, y1, x2, y2]
            box_ego_corners = np.array([
                [-1/2, -1/2],
                [-1/2, +1/2],
                [+1/2, -1/2],
                [+1/2, +1/2],
            ])*box_lws[:, np.newaxis] # [num_obstacles, 4, 2]
            cths, sths = np.cos(box_angles), np.sin(box_angles)
            rots = np.array([
                [cths, sths],
                [-sths, cths],
            ]).transpose((2, 0, 1))
            box_corners = np.matmul(
                np.expand_dims(rots.transpose(0, 2, 1), axis=1),
                box_ego_corners[:, :, :, np.newaxis]
            ).squeeze(axis=-1) + box_centers[:, np.newaxis] # [num_obstacles, 4, 2]
            segment_points = np.array([
                [box_corners[:, 0], box_corners[:, 1]],
                [box_corners[:, 0], box_corners[:, 2]],
                [box_corners[:, 3], box_corners[:, 1]],
                [box_corners[:, 3], box_corners[:, 2]],
            ]).transpose((2, 0, 1, 3)).reshape((-1, 4)) # [num_segments, 4]

            # compute segment coefficients [a, b, c] in representation ax + by + c = 0
            sp = segment_points
            segment_coefficients = np.array([
                sp[:, 1]-sp[:, 3],
                sp[:, 2]-sp[:, 0],
                sp[:, 0]*sp[:, 3]-sp[:, 2]*sp[:, 1],
            ]).transpose((1, 0)) # [num_segments, 3]

            # compute reading line coefficients in representation ax + by + c = 0
            reading_coefficients = np.array([
                -stheta,
                ctheta,
                positions[:, 0, np.newaxis]*stheta - positions[:, 1, np.newaxis]*ctheta,
            ]).transpose((1, 2, 0)) # [batch_size, num_rays, 3]

            # compute intersection points [x, y]
            sc, rc = segment_coefficients[np.newaxis, np.newaxis], reading_coefficients[:, :, np.newaxis]
            sca, scb, scc = sc[..., 0], sc[..., 1], sc[..., 2]
            rca, rcb, rcc = rc[..., 0], rc[..., 1], rc[..., 2]
            intersection_points = np.array([
                (scb*rcc-rcb*scc)/(sca*rcb-rca*scb),
                (scc*rca-rcc*sca)/(sca*rcb-rca*scb),
            ]).transpose((1, 2, 3, 0)) # [batch_size, num_rays, num_segments, 2]

            # compute intersection distances from reading positions
            ip = intersection_points
            ip_disp = ip-positions[:, np.newaxis, np.newaxis] # [batch_size, num_rays, num_segments, 2]
            rdists = np.linalg.norm(ip_disp, axis=-1) # [batch_size, num_rays, num_segments]

            # adjust reading distances for intersections in opposite direction of readings
            rdists[np.logical_or(ip_disp[..., 0]*ctheta[:, :, np.newaxis] < 0, ip_disp[..., 1]*stheta[:, :, np.newaxis] < 0)] = float('inf')

            # compute intersection distances from segment centers
            sdists = np.linalg.norm(
                ip.reshape((ip.shape[0], ip.shape[1], -1, 4, ip.shape[3]))-box_centers[np.newaxis, np.newaxis, :, np.newaxis], axis=-1
            ) # [batch_size, num_rays, num_obstacles, 4]

            # adjust reading distances for intersections not on the segments
            wllws = np.array([
                box_lws[:, 1],
                box_lws[:, 0],
                box_lws[:, 0],
                box_lws[:, 1],
            ]).transpose((1, 0))
            rdists[(sdists > (wllws/2)[np.newaxis, np.newaxis]).reshape(rdists.shape)] = float('inf')

            # filter nan
            rdists = np.nan_to_num(rdists, nan=max_distance)

            # consolidate readings for each box
            rdists = np.min(rdists, axis=-1) # [batch_size, num_rays]
            box_raydist = rdists
        else:
            box_raydist = np.full_like(thetas, float('inf'))

        # consolidate readings across obstacle types
        raydist = np.minimum(circ_raydist, box_raydist)
        error = (1-range_accuracy)*raydist # [batch_size, num_rays]
        raydist = (raydist + error*np.random.uniform(-1, 1, raydist.shape)).clip(min=min_distance, max=max_distance) # [batch_size, num_rays]
        return raydist
    
class CustomGroundEnvironment(Environment):
    def __init__(self, ground_parameter_dicts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_parameter_dicts = ground_parameter_dicts
        # ground_parameter_dicts is a list of dicts: {'x', 'y', 'length', 'width', 'friction', 'color'}