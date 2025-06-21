import pickle
import numpy as np

from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.hj_reachability_utils.dynamics import Dubins3D
from utils.hj_reachability_utils.environment import Environment
from utils.hj_reachability_utils.obstacle import CylindricalObstacle2D

# dataset parameters
datasets = {
    'training': {
        'env_dir': 'data/environments/training',
        'num_envs': 1000,
    },
    'validation': {
        'env_dir': 'data/environments/validation',
        'num_envs': 100,
    }
}

for dataset_name, parameters in datasets.items():
    print(f'generating dataset: {dataset_name}...')
    
    # create environment directory
    start_i = 0
    if not os.path.exists(parameters['env_dir']):
        os.makedirs(parameters['env_dir'])
    else:
        # continue data generation from an earlier run
        while os.path.exists(os.path.join(parameters['env_dir'], str(start_i), 'environment.pickle')):
            start_i += 1

    # generate environments
    for i in tqdm(range(start_i, parameters['num_envs'])):

        # environment parameters
        state_min = [-5, -5, -np.pi]
        state_max = [ 5,  5,  np.pi]
        state_grid_shape = [100, 100, 60]
        time_horizon = 2

        # dynamics parameters
        control_min = [0, -2]
        control_max = [2, 2]
        disturbance_norm_bounds_min = [0, 0]
        disturbance_norm_bounds_max = [1, 2]

        # obstacle parameters
        num_obstacles_min, num_obstacles_max = 1, 10
        obstacle_radius_min, obstacle_radius_max, obstacle_height = 0.1, 1, 1

        # generate obstacles
        obstacles = [CylindricalObstacle2D(
            center=[np.random.uniform(state_min[0], state_max[0]), np.random.uniform(state_min[1], state_max[1])],
            radius=np.random.uniform(obstacle_radius_min, obstacle_radius_max),
            height=obstacle_height,
        ) for _ in range(np.random.randint(num_obstacles_min, num_obstacles_max+1))]

        # generate dynamics
        dynamics = Dubins3D(
            control_min=control_min,
            control_max=control_max,
            disturbance_norm_bounds=[np.random.uniform(disturbance_norm_bounds_min[0], disturbance_norm_bounds_max[0]), np.random.uniform(disturbance_norm_bounds_min[1], disturbance_norm_bounds_max[1])],
            c1=0, c2=0, c3=0,
        )

        # generate environment
        environment = Environment(
            state_min=state_min,
            state_max=state_max,
            state_grid_shape=state_grid_shape,
            state_periodic_dims=2,
            obstacles=obstacles,
            dynamics=dynamics,
            time_horizon=time_horizon,
            progress_bar=False,
        )

        save_dir = os.path.join(parameters['env_dir'], str(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # visualize value grid
        state_slice_indices = [None, None, 0]
        environment.visualize_grid(
            grid=environment.value_grid,
            coordinate_vectors=environment.coordinate_vectors,
            axis_dims=[0, 1],
            state_slice_indices=state_slice_indices,
            save_path=os.path.join(save_dir, f'value_grid.jpg'),
            title=f'Disturbance Norm Bounds: {environment.dynamics.disturbance_norm_bounds}\nState Slice: {[np.round(environment.coordinate_vectors[j][index], 2).item() if index is not None else None for j, index in enumerate(state_slice_indices)]}',
            axis_labels=['x', 'y'],
        )

        # save environment
        with open(os.path.join(save_dir, 'environment.pickle'), 'wb') as f:
            pickle.dump(environment, f)