import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.simulation_utils.environment import Environment
from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle

# save options
num_environments = 100
save_dir = 'data/environments/simulation'

# environment options
payload_min, payload_max = -1, 1
payload_exclude = [-0.5, 0.5]
friction_min, friction_max = 0.5, 1.5
friction_exclude = [0.75, 1.25]
radius_min, radius_max = 0.1, 1
num_obstacles_min, num_obstacles_max = 4, 4
lower_left_spawn_point, upper_right_spawn_point = [4, -2], [8, 2]
min_obstacle_spacing = 0.8
obstacle_height = 1

# sampling options
sample_size = 10000
max_num_tries = 1e3

# visualization options
lower_left_vis_point, upper_right_vis_point = [-2.5, -5.5], [12.5, 5.5]
resolution = 1000
xs = np.linspace(lower_left_vis_point[0], upper_right_vis_point[0], resolution)
ys = np.linspace(lower_left_vis_point[1], upper_right_vis_point[1], resolution)
grid_xs, grid_ys = np.meshgrid(xs, ys, indexing='ij')
grid_xys = np.stack((grid_xs, grid_ys), axis=-1)

# create save directory
start_i = 0
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    # continue data generation from earlier run
    while os.path.exists(os.path.join(save_dir, str(start_i), 'environment.pickle')):
        start_i += 1

# generate environments
for i in tqdm(range(start_i, num_environments)):
    # create environment directory
    if not os.path.exists(os.path.join(save_dir, str(i))):
        os.makedirs(os.path.join(save_dir, str(i)))
    elif os.path.exists(os.path.join(save_dir, str(i), 'environment.pickle')):
        continue
    # generate obstacles
    num_obstacles = np.random.randint(low=num_obstacles_min, high=num_obstacles_max+1)
    obstacles = []
    while len(obstacles) < num_obstacles:
        obstacle_radius = np.random.uniform(low=radius_min, high=radius_max)
        num_tries = 0
        while True: # rejection sampling
            obstacle_centers = np.stack((
                np.random.uniform(low=lower_left_spawn_point[0], high=upper_right_spawn_point[0], size=sample_size),
                np.random.uniform(low=lower_left_spawn_point[1], high=upper_right_spawn_point[1], size=sample_size),
            ), axis=-1)
            min_distances = np.full(sample_size, float('inf'))
            for obstacle in obstacles:
                min_distances = np.minimum(obstacle.distances(np.array(obstacle_centers)), min_distances)
            is_valids = min_distances > obstacle_radius + min_obstacle_spacing
            if np.any(is_valids):
                break
            num_tries += 1
            if num_tries > max_num_tries:
                print('Could not create a configuration with the requested number of obstacles and spacing. Please reduce one of the quantities.')
                raise RuntimeError
        obstacles.append(CircularObstacle(center=obstacle_centers[np.argwhere(is_valids)[0].squeeze()], radius=obstacle_radius, height=obstacle_height))
    # add walls
    obstacles.append(BoxObstacle(center=[-2, 0], angle=np.pi/2, length=10, width=0.1, height=obstacle_height))
    obstacles.append(BoxObstacle(center=[12, 0], angle=np.pi/2, length=10, width=0.1, height=obstacle_height))
    obstacles.append(BoxObstacle(center=[5, -5], angle=0, length=14, width=0.1, height=obstacle_height))
    obstacles.append(BoxObstacle(center=[5, 5], angle=0, length=14, width=0.1, height=obstacle_height))
    # save environment
    payload = np.random.uniform(low=payload_min, high=payload_max)
    while payload >= payload_exclude[0] and payload <= payload_exclude[1]:
        payload = np.random.uniform(low=payload_min, high=payload_max)
    friction = np.random.uniform(low=friction_min, high=friction_max)
    while friction >= friction_exclude[0] and friction <= friction_exclude[1]:
        friction = np.random.uniform(low=friction_min, high=friction_max)
    env = Environment(
        obstacles=obstacles,
        payload=payload,
        friction=friction,
    )
    with open(os.path.join(save_dir, str(i), 'environment.pickle'), 'wb') as f:
        pickle.dump(env, f)
    # save visualization of environment
    grid_distances = env.distances(grid_xys.reshape(resolution*resolution, 2)).reshape(resolution, resolution)
    legend_limit = np.max(np.abs(grid_distances))
    plt.figure()
    plt.pcolormesh(xs, ys, grid_distances.T, cmap='RdBu', vmin=-legend_limit, vmax=legend_limit)
    plt.colorbar()
    plt.contour(xs, ys, grid_distances.T, levels=0, colors='k')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Signed Distance to Obstacles\nPayload: {env.payload:2.1f}, Friction: {env.friction:2.1f}')
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(save_dir, str(i), 'visualization.jpg'), bbox_inches='tight', dpi=800)
    plt.close()