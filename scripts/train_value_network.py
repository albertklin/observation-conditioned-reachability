import os
import torch
import wandb
import pickle

from tqdm import tqdm
from datetime import datetime
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data.data import ValueDataset
from utils.value_network.models import LiDARValueNN

# options
name = 'training'
current_time_str = datetime.now().strftime('%m_%d_%Y_%H_%M')
options = {

    # meta
    'name': name,
    'current_time_str': current_time_str,

    # save dirs
    'save_dir': os.path.join('results', name),
    'ckpt_dir': os.path.join('results', name, 'checkpoints'),
    'vis_dir': os.path.join('results', name, 'visualizations'),

    # data meta
    'train_data_dir': 'data/environments/training',
    'val_data_dir': 'data/environments/validation',
    'num_workers': 4,
    'use_cupy': False,

    # data params
    'num_repeats': 1,
    'num_rays': 100,
    'num_egos': 10,
    'ego_radius': 5,
    'num_rels': 500,
    'rel_radius': 10,
    'rel_lidar_position': [0.2, 0],

    # training meta
    'load_ckpt_path': None,
    'num_epochs': 5000,
    'num_epochs_between_ckpt': 100,
    'num_epochs_between_val': 1,

    # training params
    'batch_size': 10,
    'lr': 1e-5,
    'value_loss_weight': 1,
    'grad_loss_weight': 1,

    # visualization meta
    'num_epochs_between_vis': 100,
    'visualize_theta_index': 0,
    'num_visualizations': 10,

    # model params
    'activation': 'sine',
}

# NOTE: based on output of a commented section below, and also some rough scratchwork:
state_means = torch.tensor([0, 0, 0, 0.5, 1])
state_stds = torch.tensor([4, 4, 1.8, 0.25, 0.5])
lidar_means = 8.5*torch.ones((options['num_rays']))
lidar_stds = 3*torch.ones((options['num_rays']))
value_mean = torch.tensor([2])
value_std = torch.tensor([2])
input_means, input_stds = torch.cat((state_means, lidar_means), dim=-1), torch.cat((state_stds, lidar_stds), dim=-1)
output_mean, output_std = value_mean, value_std
options['input_means'], options['input_stds'] = input_means, input_stds
options['output_mean'], options['output_std'] = output_mean, output_std
print('input_means:', input_means)
print('input_stds:', input_stds)
print('output_mean:', output_mean)
print('output_std:', output_std)

# init wandb
print('Enter your WandB entity: ', end='')
entity = input()
wandb.init(
    project='observation-conditioned-reachability',
    entity=entity,
    group='training',
    name=options['name'],
)
wandb.config.update(options)

# create save dirs
if not os.path.exists(options['save_dir']):
    os.makedirs(options['save_dir'])
    os.makedirs(options['ckpt_dir'])
    os.makedirs(options['vis_dir'])

# save options
with open(os.path.join(options['save_dir'], 'options.pickle'), 'wb') as f:
    pickle.dump(options, f)

# create datasets
train_dataset = ValueDataset(
    options['train_data_dir'],
    num_repeats=options['num_repeats'],
    num_rays=options['num_rays'],
    num_egos=options['num_egos'],
    ego_radius=options['ego_radius'],
    num_rels=options['num_rels'],
    rel_radius=options['rel_radius'],
    rel_lidar_position=options['rel_lidar_position'],
    use_cupy=options['use_cupy'],
)
val_dataset = ValueDataset(
    options['val_data_dir'],
    num_repeats=options['num_repeats'],
    num_rays=options['num_rays'],
    num_egos=options['num_egos'],
    ego_radius=options['ego_radius'],
    num_rels=options['num_rels'],
    rel_radius=options['rel_radius'],
    use_cupy=options['use_cupy'],
)

# create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'], persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'], persistent_workers=True)

# compute data means and stds
# inputs, values, _ = next(iter(train_dataloader))
# inputs, values = inputs.flatten(0, 1), values.flatten(0, 1)
# state_means, state_stds = torch.mean(inputs[:, :5], dim=0), torch.std(inputs[:, :5], dim=0)
# lidar_mean, lidar_std = torch.mean(inputs[:, 5:]), torch.std(inputs[:, 5:])
# value_mean, value_std = torch.mean(values), torch.std(values)
# print(state_means, state_stds)
# print(lidar_mean, lidar_std)
# print(value_mean, value_std)

# create model
model = LiDARValueNN(
    input_means.cuda(), input_stds.cuda(),
    output_mean.cuda(), output_std.cuda(),
    input_dim=5+options['num_rays'],
    activation=options['activation'],
).cuda()
if options['load_ckpt_path'] is not None:
    model.load_state_dict(torch.load(options['load_ckpt_path']))
model.train()

# create training objects
mseloss = MSELoss()
opt = Adam(model.parameters(), lr=options['lr'])

# training loop
best_val_loss = float('inf')
for epoch in tqdm(range(options['num_epochs'])):
    for i, data in tqdm(enumerate(train_dataloader), leave=False):
        inputs, values, grads = data
        inputs, values, grads = torch.flatten(inputs, end_dim=1).float().cuda(), torch.flatten(values, end_dim=1).float().cuda(), torch.flatten(grads, end_dim=1).float().cuda()

        # create states leaf tensor to allow grad computation
        states = inputs[:, :3].detach().clone().requires_grad_(True)
        inputs[:, :3] = states

        # compute loss
        opt.zero_grad()
        pred_values = model.forward(inputs)
        pred_grads = torch.autograd.grad(pred_values.unsqueeze(-1), states, torch.ones_like(pred_values.unsqueeze(-1)), create_graph=True)[0]
        value_loss = mseloss(pred_values, values)
        grad_loss = mseloss(pred_grads, grads)
        total_loss = options['value_loss_weight']*value_loss + options['grad_loss_weight']*grad_loss
        total_loss.backward()
        opt.step()
        wandb.log({
            'epoch': epoch + (i+1)/len(train_dataloader),
            'train_batch_value_loss': value_loss.item(),
            'train_batch_grad_loss': grad_loss.item(),
            'train_batch_total_loss': total_loss.item(),
        })

    # save checkpoint
    if (epoch+1)%options['num_epochs_between_ckpt'] == 0:
        torch.save(model.state_dict(), os.path.join(options['ckpt_dir'], f'epoch_{str(epoch+1).zfill(len(str(options["num_epochs"])))}.pth'))
    if (epoch+1) == options['num_epochs']:
        torch.save(model.state_dict(), os.path.join(options['ckpt_dir'], f'final_model.pth'))

    # validate model
    if (epoch+1)%options['num_epochs_between_val'] == 0:

        # set model to eval and freeze
        model.eval()
        param_requires_grad_dict = {}
        for param in model.parameters():
            param_requires_grad_dict[param] = param.requires_grad
            param.requires_grad = False

        inputs, values, grads = next(iter(val_dataloader))
        inputs, values, grads = torch.flatten(inputs, end_dim=1).float().cuda(), torch.flatten(values, end_dim=1).float().cuda(), torch.flatten(grads, end_dim=1).float().cuda()
        
        # create states leaf tensor to allow grad computation
        states = inputs[:, :3].detach().clone().requires_grad_(True)
        inputs[:, :3] = states

        # compute loss
        pred_values = model.forward(inputs)
        pred_grads = torch.autograd.grad(pred_values.unsqueeze(-1), states, torch.ones_like(pred_values.unsqueeze(-1)), create_graph=False)[0]
        value_loss = mseloss(pred_values, values)
        grad_loss = mseloss(pred_grads, grads)
        total_loss = options['value_loss_weight']*value_loss + options['grad_loss_weight']*grad_loss
        wandb.log({
            'epoch': epoch+1,
            'val_batch_value_loss': value_loss.item(),
            'val_batch_grad_loss': grad_loss.item(),
            'val_batch_total_loss': total_loss.item(),
        })

        # set model to train and unfreeze
        model.train()
        for param in model.parameters():
            param.requires_grad = param_requires_grad_dict[param]

        # save best val checkpoint
        if total_loss.item() < best_val_loss:
            best_val_loss = total_loss.item()
            torch.save(model.state_dict(), os.path.join(options['ckpt_dir'], f'best_val.pth'))
    
    # visualize model
    if (epoch+1)%options['num_epochs_between_vis'] == 0:
        # TODO: implement
        pass
