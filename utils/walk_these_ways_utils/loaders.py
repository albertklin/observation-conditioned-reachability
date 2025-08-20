import glob
import torch
import pickle as pkl
import numpy as np

from isaacgym import gymtorch, gymapi, gymutil, torch_utils
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis

from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

from utils.hj_reachability_utils.obstacle import CylindricalObstacle2D
from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle

VIS_LINE_Z = 0.01 # slightly above ground to ensure no obscurring issues

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, task, payload, friction, headless=False, body_color=(1, 1, 1)):
    dirs = glob.glob(f"libraries/walk-these-ways/runs/{label}/*")
    logdir = sorted(dirs)[0]

    policy = load_policy(logdir)

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key) and key != 'command_ranges':
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR and early termination for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_roll_pitch = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.env.episode_length_s = 3600
    Cfg.terrain.mesh_type = 'plane'
    Cfg.terrain.x_init_range = 0
    Cfg.terrain.y_init_range = 0
    Cfg.terrain.yaw_init_range = 0
    Cfg.terrain.x_init_offset = 0
    Cfg.terrain.y_init_offset = 0

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    # set payload
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = payload, payload

    # set friction
    Cfg.terrain.static_friction = -1 + friction
    Cfg.terrain.dynamic_friction = -1 + friction

    env = VelocityTrackingEasyEnv(sim_device=f'cuda:{torch.cuda.current_device()}', headless=headless, cfg=Cfg, task=task)
    env = HistoryWrapper(env)

    body_rigid_shape_indices = [0]
    for i in body_rigid_shape_indices: # set body color for relevant rigid shapes
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*body_color))

    # custom helper functions for adding and refreshing drawings
    env.perm_lines = []
    env.temp_lines = []
    def add_line_to_list(list, x1, y1, x2, y2, c):
        list.append(((x1, y1, VIS_LINE_Z), (x2, y2, VIS_LINE_Z), c))
    def add_perm_line(x1, y1, x2, y2, c):
        add_line_to_list(env.perm_lines, x1, y1, x2, y2, c)
    def add_temp_line(x1, y1, x2, y2, c):
        add_line_to_list(env.temp_lines, x1, y1, x2, y2, c)
    def add_circ_to_list(list, x, y, r, n, c):
        ts = np.linspace(0, 2*np.pi, num=n)
        xs = r*np.cos(ts) + x
        ys = r*np.sin(ts) + y
        for i in range(n-1):
            add_line_to_list(list, xs[i], ys[i], xs[i+1], ys[i+1], c)
    def add_perm_circ(x, y, r, n, c):
        add_circ_to_list(env.perm_lines, x, y, r, n, c)
    def add_temp_circ(x, y, r, n, c):
        add_circ_to_list(env.temp_lines, x, y, r, n, c)
    def add_box_to_list(list, x, y, th, l, w, c):
        ego_corners = np.array([
            [-l/2, -w/2],
            [-l/2, +w/2],
            [+l/2, -w/2],
            [+l/2, +w/2],
        ])
        cth, sth = np.cos(th), np.sin(th)
        rot = np.array([
            [cth, sth],
            [-sth, cth],
        ])
        world_corners = np.matmul(rot.T, ego_corners[:, :, np.newaxis]).squeeze(axis=-1) + np.array([x, y])
        add_line_to_list(list, *world_corners[0], *world_corners[1], c)
        add_line_to_list(list, *world_corners[0], *world_corners[2], c)
        add_line_to_list(list, *world_corners[3], *world_corners[1], c)
        add_line_to_list(list, *world_corners[3], *world_corners[2], c)
    def add_perm_box(x, y, th, l, w, c):
        add_box_to_list(env.perm_lines, x, y, th, l, w, c)
    def add_temp_box(x, y, th, l, w, c):
        add_box_to_list(env.temp_lines, x, y, th, l, w, c)
    def remove_temp_lines():
        env.temp_lines = []
    def refresh_drawings():
        env.gym.clear_lines(env.viewer)
        def draw_lines(lines):
            if len(lines) == 0:
                return
            ls = np.empty((len(lines), 2), gymapi.Vec3.dtype)
            cs = np.empty(len(lines), gymapi.Vec3.dtype)
            for i in range(len(lines)):
                ls[i][0] = lines[i][0]
                ls[i][1] = lines[i][1]
                cs[i] = lines[i][2]
            env.gym.add_lines(env.viewer, env.envs[0], len(lines), ls, cs)
        draw_lines(env.perm_lines)
        draw_lines(env.temp_lines)

    # add goal position
    add_perm_circ(task.goal_position[0], task.goal_position[1], task.goal_radius, 100, (0, 1, 0))
    # add obstacles
    obstacle_color = (1, 0, 0)
    for obstacle in task.environment.obstacles:
        if isinstance(obstacle, CylindricalObstacle2D) or isinstance(obstacle, CircularObstacle):
            add_perm_circ(obstacle.center[0], obstacle.center[1], obstacle.radius, 100, obstacle_color)
        elif isinstance(obstacle, BoxObstacle):
            add_perm_box(*obstacle.center, obstacle.angle, obstacle.length, obstacle.width, obstacle_color)
        else:
            raise NotImplementedError
    # refresh drawings
    refresh_drawings()

    # custom helper function to set robot state
    def set_robot_state(x, y, th):
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.root_states[env_ids] = env.base_init_state
        env.root_states[env_ids, :3] += env.env_origins[env_ids]
        env.root_states[env_ids, :2] = torch.Tensor([x, y]).to(env.device)

        # base yaws
        init_yaws = torch_rand_float(th,
                                     th, (len(env_ids), 1),
                                     device=env.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(env.device))[:, 0, :]
        env.root_states[env_ids, 3:7] = quat

        # base velocities
        env.root_states[env_ids, 7:13] = torch_rand_float(0, 0, (len(env_ids), 6),
                                                           device=env.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                     gymtorch.unwrap_tensor(env.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # custom helper function to step in environment using Dubins3D action
    def step_dubins3d(v, w, obs, add_line=True, line_color=(0, 0, 1), body_color=(1, 1, 1)):
        prev_x, prev_y = env.base_pos[0, 0].item(), env.base_pos[0, 1].item()

        gaits = {"pronking": [0, 0, 0],
                 "trotting": [0.5, 0, 0],
                 "bounding": [0, 0.5, 0],
                 "pacing": [0, 0, 0.5]}

        y_vel_cmd = 0.0
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25

        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = v
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = w
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        if add_line:
            curr_x, curr_y = env.base_pos[0, 0].item(), env.base_pos[0, 1].item()
            add_perm_line(prev_x, prev_y, curr_x, curr_y, line_color)

        for i in body_rigid_shape_indices: # set body color for relevant rigid shapes
            env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*body_color))
            
        return obs
    
    # custom helper function to get the current Dubins3D state
    def current_dubins3d_state():
        return np.array([
            env.base_pos[0, 0].item(),
            env.base_pos[0, 1].item(),
            torch_utils.get_euler_xyz(env.base_quat)[2][0].item(),
        ])
    
    # custom helper function to check for collision
    def in_collision():
        return torch.any(torch.linalg.norm(gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim))[-len(task.environment.obstacles):], dim=-1) > 0)
    
    # expose useful helper functions
    env.set_robot_state = set_robot_state
    env.step_dubins3d = step_dubins3d
    env.current_dubins3d_state = current_dubins3d_state
    env.in_collision = in_collision
    env.add_temp_line = add_temp_line
    env.add_temp_circ = add_temp_circ
    env.add_temp_box = add_temp_box
    env.remove_temp_lines = remove_temp_lines
    env.refresh_drawings = refresh_drawings

    return env