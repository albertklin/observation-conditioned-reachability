import os
import pickle
# import cupy
import numpy as np
from torch.utils.data import Dataset

class ValueDataset(Dataset):
    """A value dataset with input (x, y, th, dst_dxdy_max, dst_dth_max, lidar) and label (value, grad)."""

    def __init__(
            self, 
            dir: str,
            num_repeats: int = 10,
            num_rays: int = 100,
            num_egos: int = 10,
            ego_radius: float = 4,
            num_rels: int = 100,
            rel_radius: float = 1,
            rel_lidar_position: "list[float]" = [0, 0],
            num_ego_tries: int = int(1e12),
            ego_batch_size: int = 0,
            num_rel_tries: int = int(1e12),
            rel_batch_size: int = 0,
            use_cupy: bool = False,
    ):
        """Initializes a value dataset.
        
        Args:
            dir: The dataset directory with environments.
            num_repeats: The number of times the dataset is repeated to artificially increase its length (to amortize the constant overhead for each enumeration).
            num_rays: The number of LiDAR rays.
            num_egos: The number of ego states to sample per environment.
            ego_radius: The radius of the sampling space for ego states.
            num_rels: The number of relative states to sample per ego state.
            rel_radius: The radius of the sampling space for the relative states.
            rel_lidar_position: The relative position of the lidar.
            num_ego_tries: The maximum number of times to try sampling for ego states, to prevent sampling indefinitely.
            ego_batch_size: The batch size for sampling ego states.
            num_rel_tries: The maximum number of times to try sampling for rel states, to prevent sampling indefinitely.
            rel_batch_size: The batch size for sampling rel states.
            use_cupy: Whether to use cupy for GPU acceleration. For the current implementation, there is actually NO improvement in speed.
        """
        self.dir = dir
        self.num_repeats = num_repeats
        self.num_rays = num_rays
        self.num_egos = num_egos
        self.ego_radius = ego_radius
        self.num_rels = num_rels
        self.rel_radius = rel_radius
        self.rel_lidar_position = rel_lidar_position
        self.num_ego_tries = num_ego_tries
        self.ego_batch_size = ego_batch_size if ego_batch_size > 0 else 2*num_egos
        self.num_rel_tries = num_rel_tries
        self.rel_batch_size = rel_batch_size if rel_batch_size > 0 else 10*num_rels
        self.use_cupy = use_cupy

        self.num_envs = len(os.listdir(self.dir))
        self.norm_rel_lidar_position = np.linalg.norm(rel_lidar_position).item()
            
    def __len__(self):
        return self.num_repeats*self.num_envs
    
    def __getitem__(self, idx):
        """Returns a sample of data from environment idx.
        Contains num_egos*num_rels input-label pairs.
        NOTE: If there are insufficient states to sample, it will return the incomplete set it is able to find.
        
        Returns:
            (inputs, value_labels, grad_labels)
            where:
                inputs: A numpy array with shape [num_egos * num_rels, state_dim + disturbance_bound_dim + num_rays].
                value_labels: A numpy array with shape [num_egos * num_rels].
                grad_labels: A numpy array with shape [num_egos * num_rels, state_dim]
        """
        # lib = cupy if self.use_cupy else np
        lib = NotImplementedError if self.use_cupy else np

        # load env
        with open(os.path.join(self.dir, str(idx%self.num_envs), 'environment.pickle'), 'rb') as f:
            env = pickle.load(f)
        coordinate_vectors = tuple(lib.asarray(cv) for cv in env.coordinate_vectors)
        state_grid = lib.asarray(env.state_grid)
        value_grid = lib.asarray(env.value_grid)
        grad_grid = lib.asarray(env.grad_grid)
        distance_grid = lib.asarray(env.distance_grid)
        disturbance_bound = lib.asarray(env.dynamics.disturbance_norm_bounds)
        
        # sample ego states
        ego_indices = None
        candidate_ego_x_indices = lib.nonzero(lib.abs(coordinate_vectors[0]) < self.ego_radius)[0]
        candidate_ego_y_indices = lib.nonzero(lib.abs(coordinate_vectors[1]) < self.ego_radius)[0]
        candidate_ego_th_indices = lib.arange(len(coordinate_vectors[2]))
        for _ in range(self.num_ego_tries):
            sample_ego_indices = (
                lib.random.choice(candidate_ego_x_indices, self.ego_batch_size, replace=True),
                lib.random.choice(candidate_ego_y_indices, self.ego_batch_size, replace=True),
                lib.random.choice(candidate_ego_th_indices, self.ego_batch_size, replace=True),
            )
            valid_lidars = distance_grid[sample_ego_indices] > self.norm_rel_lidar_position # heuristic
            if ego_indices is None:
                ego_indices = tuple(indices[valid_lidars] for indices in sample_ego_indices)
            else:
                ego_indices = tuple(lib.concatenate((ego_indices[i], sample_ego_indices[i][valid_lidars])) for i in range(len(ego_indices)))
            if len(ego_indices[0]) >= self.num_egos:
                ego_indices = tuple(indices[:self.num_egos] for indices in ego_indices)
                break
        ego_states = state_grid[ego_indices]
        
        # compute LiDAR positions in world frame
        lidar_positions = ego_states[:, :2].copy()
        cths, sths = lib.cos(ego_states[:, 2]), lib.sin(ego_states[:, 2])
        lidar_positions[:, 0] = lidar_positions[:, 0] + self.rel_lidar_position[0]*cths - self.rel_lidar_position[1]*sths
        lidar_positions[:, 1] = lidar_positions[:, 1] + self.rel_lidar_position[0]*sths + self.rel_lidar_position[1]*cths

        # compute LiDAR readings
        thetas = ego_states[:, 2:3] + lib.linspace(-lib.pi, lib.pi, self.num_rays, endpoint=False)[lib.newaxis]
        lidars = env.read_lidar(
            lidar_positions,
            thetas,
            use_cupy=self.use_cupy,
        )

        # sample relative states and accumulate input-label pairs
        inputs, value_labels, grad_labels = [], [], []
        for i in range(len(ego_states)):
            # sample relative states
            rel_indices = None
            candidate_rel_x_indices = lib.nonzero(lib.abs(coordinate_vectors[0]-ego_states[i, 0]) < self.rel_radius)[0]
            candidate_rel_y_indices = lib.nonzero(lib.abs(coordinate_vectors[1]-ego_states[i, 1]) < self.rel_radius)[0]
            candidate_rel_th_indices = lib.arange(len(coordinate_vectors[2]))
            for _ in range(self.num_rel_tries):
                sample_rel_indices = (
                    lib.random.choice(candidate_rel_x_indices, self.rel_batch_size, replace=True),
                    lib.random.choice(candidate_rel_y_indices, self.rel_batch_size, replace=True),
                    lib.random.choice(candidate_rel_th_indices, self.rel_batch_size, replace=True),
                )
                sample_rel_states = state_grid[sample_rel_indices]
                sample_angles = lib.arctan2(sample_rel_states[:, 1]-lidar_positions[i, 1], sample_rel_states[:, 0]-lidar_positions[i, 0])
                sample_lidars = lib.interp(sample_angles, thetas[i], lidars[i], period=2*lib.pi)
                observable_states = lib.linalg.norm(sample_rel_states[:, :2]-lidar_positions[i], axis=-1) <= sample_lidars
                if rel_indices is None:
                    rel_indices = tuple(indices[observable_states] for indices in sample_rel_indices)
                else:
                    rel_indices = tuple(lib.concatenate((rel_indices[i], sample_rel_indices[i][observable_states])) for i in range(len(rel_indices)))
                if len(rel_indices[0]) >= self.num_rels:
                    rel_indices = tuple(indices[:self.num_rels] for indices in rel_indices)
                    break
            rel_states = state_grid[rel_indices]
            rel_values = value_grid[rel_indices]
            rel_grads = grad_grid[rel_indices]

            # compute relative states and grads in ego frame
            cth, sth = cths[i], sths[i]
            rot = lib.array([
                [cth, sth],
                [-sth, cth],
            ])
            rel_states[:, :2] = lib.matmul(rot, (rel_states[:, :2] - ego_states[i, :2])[:, :, lib.newaxis]).squeeze(axis=-1)
            rel_grads[:, :2] = lib.matmul(rot, rel_grads[:, :2, lib.newaxis]).squeeze(axis=-1)
            rel_states[:, 2] = rel_states[:, 2] - ego_states[i, 2]
            rel_states = env.wrap_states(rel_states, use_cupy=self.use_cupy)
            
            # construct ego infos
            ego_infos = lib.ones((len(rel_states), 1))*lib.concatenate(
                (disturbance_bound, lidars[i]), axis=-1
            )

            # store input-label pairs
            inputs.append(lib.concatenate((rel_states, ego_infos), axis=-1))
            value_labels.append(rel_values)
            grad_labels.append(rel_grads)

            # # plot world frame and then relative states and LiDAR readings in ego frame just to confirm...
            # import matplotlib.pyplot as plt
            # rel_x, rel_y, rel_th = rel_states[:, 0], rel_states[:, 1], rel_states[:, 2]
            # pos = lib.matmul(rot.T, lib.stack((rel_x, rel_y), axis=-1)[:, :, lib.newaxis]).squeeze(axis=-1)  + ego_states[i, :2]
            # x, y, th = pos[:, 0], pos[:, 1], rel_th + ego_states[i, 2]
            # v = rel_values
            # fig, axs = plt.subplots(2, 2, figsize=(24, 24))
            # axs[0, 0].set_aspect('equal')
            # axs[0, 0].set_xlim(-15, 15)
            # axs[0, 0].set_ylim(-15, 15)
            # axs[0, 0].set_xlabel('x')
            # axs[0, 0].set_ylabel('y')
            # axs[0, 0].set_title('Occupancy and Simulated LiDAR Readings')
            
            # axs[0, 0].pcolormesh(coordinate_vectors[0],
            #                 coordinate_vectors[1],
            #                 distance_grid[:, :, 0].T,
            #                 cmap='RdBu',
            #                 vmin=-1, vmax=1)
            # axs[0, 0].contour(coordinate_vectors[0],
            #             coordinate_vectors[1],
            #             distance_grid[:, :, 0].T,
            #             levels=0,
            #             colors='black')
            
            # # abs
            # axs[0, 0].scatter(ego_states[i, 0], ego_states[i, 1])
            # axs[0, 0].quiver(ego_states[i, 0], ego_states[i, 1], lib.cos(ego_states[i, 2]), lib.sin(ego_states[i, 2]), angles='xy')
            # axs[0, 0].scatter(lidars[i]*lib.cos(thetas[i]) + ego_states[i, 0], lidars[i]*lib.sin(thetas[i]) + ego_states[i, 1])
            # axs[0, 0].quiver(x, y, np.cos(th), np.sin(th), angles='xy', width=0.001, scale=20)
            # axs[0, 0].scatter(x[v<0], y[v<0], c='r', s=1)
            # axs[0, 0].scatter(x[v>0], y[v>0], c='b', s=1)
            
            # # rel
            # axs[1, 0].set_aspect('equal'), axs[1, 1].set_aspect('equal')
            # axs[1, 0].plot(lidars[i]*lib.cos(thetas[i]-ego_states[i, 2]), lidars[i]*lib.sin(thetas[i]-ego_states[i, 2]), markersize=2, marker='*', color='r')
            # axs[1, 0].quiver(rel_x[v<0], rel_y[v<0], np.cos(rel_th[v<0]), np.sin(rel_th[v<0]), angles='xy', width=0.001)
            # axs[1, 0].scatter(rel_x[v<0], rel_y[v<0], c='r')
            # axs[1, 1].set_title('unsafe')
            # axs[1, 1].plot(lidars[i]*lib.cos(thetas[i]-ego_states[i, 2]), lidars[i]*lib.sin(thetas[i]-ego_states[i, 2]), markersize=2, marker='*', color='r')
            # axs[1, 1].quiver(rel_x[v>0], rel_y[v>0], np.cos(rel_th[v>0]), np.sin(rel_th[v>0]), angles='xy', width=0.001)
            # axs[1, 1].scatter(rel_x[v>0], rel_y[v>0], c='b')
            # axs[1, 1].set_title('safe')
            # fig.suptitle(ego_infos[0, :2])
            # plt.show()

        inputs = lib.concatenate(inputs, axis=0)
        value_labels = lib.concatenate(value_labels, axis=0)
        grad_labels = lib.concatenate(grad_labels, axis=0)
        if len(inputs) < self.num_egos*self.num_rels:
            print(f'WARNING: missing data (collected only {len(inputs)}/{self.num_egos*self.num_rels})...')

        if self.use_cupy:
            inputs, value_labels, grad_labels = inputs.get(), value_labels.get(), grad_labels.get()

        return inputs, value_labels, grad_labels