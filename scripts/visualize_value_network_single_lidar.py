import os
# import cupy
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.value_network.models import LiDARValueNN

env_dir = 'data/environments/validation/0'
ckpt_path = 'results/training/checkpoints/epoch_05000.pth'
options_path = 'results/training/options.pickle'
visualize_theta_index = 0

with open(options_path, 'rb') as f:
    options = pickle.load(f)
rel_lidar_position = options['rel_lidar_position']

# lib = cupy if options['use_cupy'] else np
lib = NotImplementedError if options['use_cupy'] else np

# load env
with open(os.path.join(env_dir, 'environment.pickle'), 'rb') as f:
    env = pickle.load(f)

# load ckpt
model = LiDARValueNN(
    options['input_means'].cuda(), options['input_stds'].cuda(),
    options['output_mean'].cuda(), options['output_std'].cuda(),
    input_dim=5+options['num_rays'],
    activation=options['activation']
).cuda()
model.load_state_dict(torch.load(ckpt_path))
model.eval()
for param in model.parameters():
    param.requires_grad = False

# true grids
coordinate_vectors = [lib.asarray(cv) for cv in env.coordinate_vectors]
theta = coordinate_vectors[2][visualize_theta_index].item()
disturbance_bound = lib.asarray(env.dynamics.disturbance_norm_bounds)
value_grid = lib.asarray(env.value_grid).copy()
grad_grid = lib.asarray(env.grad_grid).copy()

# compute single LiDAR
state = lib.array([0, 0, theta])
# compute LiDAR position
lidar_position = state[:2].copy()
cth, sth = lib.cos(state[2]), lib.sin(state[2])
lidar_position[0] = lidar_position[0] + rel_lidar_position[0]*cth - rel_lidar_position[1]*sth
lidar_position[1] = lidar_position[1] + rel_lidar_position[0]*sth + rel_lidar_position[1]*cth
thetas = state[2] + lib.linspace(-lib.pi, lib.pi, options['num_rays'], endpoint=False)
lidar = env.read_lidar(
    lidar_position[lib.newaxis],
    thetas[lib.newaxis],
    use_cupy=options['use_cupy'],
)[0]

# predict grids
grid_xs, grid_ys = lib.meshgrid(env.coordinate_vectors[0], env.coordinate_vectors[1], indexing='ij')
states = lib.stack((grid_xs.flatten(), grid_ys.flatten(), theta*lib.ones((len(env.coordinate_vectors[0])*len(env.coordinate_vectors[1])))), axis=-1)
rot = lib.array([
    [cth, sth],
    [-sth, cth],
])
rel_states = states.copy()
rel_states[:, :2] = lib.matmul(rot, (rel_states[:, :2] - state[:2])[:, :, lib.newaxis]).squeeze(axis=-1)
rel_states[:, 2] = rel_states[:, 2] - state[2]
rel_states[:, 2] = (rel_states[:, 2] + lib.pi)%(2*lib.pi) - lib.pi # wrap to [-pi, pi]
inputs = lib.concatenate((rel_states, disturbance_bound*lib.ones((len(states), 1)), lidar*lib.ones((len(states), 1))), axis=-1)
inputs = torch.as_tensor(inputs, dtype=torch.float32, device='cuda')
# create input states leaf tensor to allow grad computation
input_states = inputs[:, :3].detach().clone().requires_grad_(True)
inputs[:, :3] = input_states
# forward pass
pred_values = model.forward(inputs)
pred_grads = torch.autograd.grad(pred_values.unsqueeze(-1), input_states, torch.ones_like(pred_values.unsqueeze(-1)), create_graph=False)[0]
# NOTE: These are grads w.r.t. the ego frame, which needs to be transformed before plotting, since the plot is in the world frame.
cth, sth = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
rot = torch.tensor([
    [cth, sth],
    [-sth, cth],
], device='cuda')
pred_grads[:, :2] = torch.matmul(rot.T, pred_grads[:, :2, None]).squeeze(-1)
pred_value_grid = pred_values.reshape(*value_grid.shape[:2]).detach().cpu().numpy()
pred_grad_grid = pred_grads.reshape(*grad_grid.shape[:2], grad_grid.shape[3]).detach().cpu().numpy()

# # erase beyond LiDAR
# state_angles = lib.arctan2(states[:, 1]-lidar_position[1], states[:, 0]-lidar_position[0])
# state_lidars = lib.interp(state_angles, thetas, lidar, period=2*lib.pi)
# observable_states = lib.linalg.norm(states[:, :2]-lidar_position, axis=-1) <= state_lidars
# observable_states_grid = observable_states.reshape(*value_grid.shape[:2])
# value_grid[lib.logical_not(observable_states_grid)] = lib.nan
# grad_grid[lib.logical_not(observable_states_grid)] = lib.nan
# pred_value_grid[lib.logical_not(observable_states_grid)] = lib.nan
# pred_grad_grid[lib.logical_not(observable_states_grid)] = lib.nan

# plot true/pred value/grad grids
fig, axs = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle(f'Model: {ckpt_path}\nEnvironment: {env_dir}\nDisturbance bound: {disturbance_bound}\n$\\theta={theta}$')
for axs_r in axs:
    for ax in axs_r:
        ax.set_aspect('equal'), ax.set_xlabel('x'), ax.set_ylabel('y')

# true value grid
legend_limit = 6
axs[0, 0].set_title('True Value Grid')
axs[0, 0].pcolormesh(
    coordinate_vectors[0],
    coordinate_vectors[1],
    value_grid[:, :, visualize_theta_index].T,
    cmap='RdBu',
    vmin=-legend_limit, vmax=legend_limit
)
# axs[0, 0].contour(
#     coordinate_vectors[0],
#     coordinate_vectors[1],
#     value_grid[:, :, visualize_theta_index].T,
#     levels=0,
#     colors='black',
#     linewidths=0.4,
# )
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-legend_limit, vmax=legend_limit), cmap='RdBu'), ax=axs[0, 0], fraction=0.046, pad=0.04)

# true grad grids
legend_limit = 2
state_labels = ['$x$', '$y$', '$\\theta$']
for i in range(grad_grid.shape[-1]):
    axs[0, i+1].set_title(f'True Grad Grid w.r.t. {state_labels[i]}')
    axs[0, i+1].pcolormesh(
        coordinate_vectors[0],
        coordinate_vectors[1],
        grad_grid[:, :, visualize_theta_index, i].T,
        cmap='RdBu',
        vmin=-legend_limit, vmax=legend_limit
    )
    # axs[0, i+1].contour(
    #     coordinate_vectors[0],
    #     coordinate_vectors[1],
    #     grad_grid[:, :, visualize_theta_index, i].T,
    #     levels=0,
    #     colors='black',
    #     linewidths=0.4,
    # )
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-legend_limit, vmax=legend_limit), cmap='RdBu'), ax=axs[0, i+1], fraction=0.046, pad=0.04)

# pred value grid
legend_limit = 6
axs[1, 0].set_title('Predicted Value Grid')
axs[1, 0].pcolormesh(
    coordinate_vectors[0],
    coordinate_vectors[1],
    pred_value_grid.T,
    cmap='RdBu',
    vmin=-legend_limit, vmax=legend_limit
)
# axs[1, 0].contour(
#     coordinate_vectors[0],
#     coordinate_vectors[1],
#     pred_value_grid.T,
#     levels=0,
#     colors='black',
#     linewidths=0.4,
# )
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-legend_limit, vmax=legend_limit), cmap='RdBu'), ax=axs[1, 0], fraction=0.046, pad=0.04)

# pred grad grids
legend_limit = 2
state_labels = ['$x$', '$y$', '$\\theta$']
for i in range(pred_grad_grid.shape[-1]):
    axs[1, i+1].set_title(f'Predicted Grad Grid w.r.t. {state_labels[i]}')
    axs[1, i+1].pcolormesh(
        coordinate_vectors[0],
        coordinate_vectors[1],
        pred_grad_grid[:, :, i].T,
        cmap='RdBu',
        vmin=-legend_limit, vmax=legend_limit
    )
    # axs[1, i+1].contour(
    #     coordinate_vectors[0],
    #     coordinate_vectors[1],
    #     pred_grad_grid[:, :, i].T,
    #     levels=0,
    #     colors='black',
    #     linewidths=0.4,
    # )
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-legend_limit, vmax=legend_limit), cmap='RdBu'), ax=axs[1, i+1], fraction=0.046, pad=0.04)

# plt.gcf().set_dpi(800)
# plt.tight_layout()
plt.savefig('value_network_prediction_single_lidar.jpg', dpi=600)
