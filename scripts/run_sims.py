import os
import sys
import time
import shutil
import math
import pickle
import traceback
import isaacgym
import torch
assert isaacgym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from isaacgym import gymapi
from scipy import interpolate
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import RPLidarA1 as LaserModel # for sim, RPLidarA1 is the same as RPLidarA2
from skimage.measure import block_reduce

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.walk_these_ways_utils.loaders import load_env
from utils.navigation_task import NavigationTask
from utils.dynamics import Dubins3D
from utils.predictive_sampler import PredictiveSampler
from utils.disturbance_estimator import DisturbanceEstimator
from utils.value_network.models import LiDARValueNN
from utils.online_environment import OnlineEnvironment
from utils.safety_filter import SafetyFilter

os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/nvidia_icd.json'

SAFE_COLOR = (92/255, 128/255, 64/255)
UNSAFE_COLOR = (128/255, 64/255, 64/255)

SAVE_DIR = 'results/simulation'
ENV_DIR = 'data/environments/simulation'
CKPT_PATH = 'results/training/checkpoints/epoch_05000.pth'
OPTIONS_PATH = 'results/training/options.pickle'
HEADLESS = False

# load options
with open(OPTIONS_PATH, 'rb') as f:
    options = pickle.load(f)

# load ckpt
model = LiDARValueNN(
    options['input_means'].cuda(), options['input_stds'].cuda(),
    options['output_mean'].cuda(), options['output_std'].cuda(),
    input_dim=5+options['num_rays'],
).cuda()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()
for param in model.parameters():
    param.requires_grad = False

if not HEADLESS:
    # set up SLAM plot
    plt.ion()
    plt.figure()
    plt.title('SLAM and MPS Planning')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().set_aspect('equal')

# run simulations
for env_id in tqdm(os.listdir(ENV_DIR)):
    
    # create env-specific save dir
    env_save_dir = os.path.join(SAVE_DIR, env_id)
    if not os.path.exists(env_save_dir):
        os.makedirs(env_save_dir)
    else:
        if os.path.exists(os.path.join(SAVE_DIR, env_id, 'data.pickle')):
            # skip, since simulation data has already been generated for this env
            continue
        # simulation was interrupted, so start from scratch
        shutil.rmtree(env_save_dir)
        os.makedirs(env_save_dir)

    # load env
    with open(os.path.join(ENV_DIR, env_id, 'environment.pickle'), 'rb') as f:
        env = pickle.load(f)

    # SLAM parameters
    SLAM_MAP_SIZE_PIXELS = 2000
    SLAM_REDUCED_MAP_SIZE_PIXELS = 200
    assert (SLAM_MAP_SIZE_PIXELS//SLAM_REDUCED_MAP_SIZE_PIXELS)*SLAM_REDUCED_MAP_SIZE_PIXELS == SLAM_MAP_SIZE_PIXELS
    SLAM_MAP_SIZE_METERS = 20
    SLAM_X = np.linspace(-SLAM_MAP_SIZE_METERS/2, SLAM_MAP_SIZE_METERS/2, SLAM_MAP_SIZE_PIXELS)
    SLAM_Y = np.linspace(-SLAM_MAP_SIZE_METERS/2, SLAM_MAP_SIZE_METERS/2, SLAM_MAP_SIZE_PIXELS)
    SLAM_REDUCED_X = np.linspace(-SLAM_MAP_SIZE_METERS/2, SLAM_MAP_SIZE_METERS/2, SLAM_REDUCED_MAP_SIZE_PIXELS)
    SLAM_REDUCED_Y = np.linspace(-SLAM_MAP_SIZE_METERS/2, SLAM_MAP_SIZE_METERS/2, SLAM_REDUCED_MAP_SIZE_PIXELS)
    SLAM_OCCUPANCY_THRESHOLD = 0.85
    SLAM_EMPTY_THRESHOLD = 0.01

    # set up SLAM environment
    slam_environment = OnlineEnvironment(
        X=SLAM_REDUCED_X,
        Y=SLAM_REDUCED_Y,
        TH=np.array([-np.pi, np.pi]),
    )
    
    # set up task
    timeout = 60
    dynamics = Dubins3D()
    initial_position = np.array([0, 0, 0])
    task = NavigationTask(
        robot_radius=0.35,
        goal_position=np.array([10, 0]),
        goal_radius=0.5,
        environment=env,
        dynamics=dynamics
    )
    online_task = NavigationTask(
        robot_radius=0.35,
        goal_position=np.array([10, 0]),
        goal_radius=0.5,
        environment=slam_environment,
        dynamics=dynamics
    )

    # create sim
    sim_env = load_env(
        "gait-conditioned-agility/pretrain-v0/train", task,
        payload=env.payload, friction=env.friction, headless=HEADLESS,
        body_color=SAFE_COLOR
    )

    # set camera angle
    sim_env.set_camera([5/2, 0, 11], [5/2, 1e-12, 0])

    # set lighting parameters
    sim_env.gym.set_light_parameters(
        sim_env.sim, 0,
        gymapi.Vec3(0.8, 0.8, 0.8),
        gymapi.Vec3(0.8, 0.8, 0.8),
        gymapi.Vec3(-0.5, 0, -1),
    )
    sim_env.gym.set_light_parameters(
        sim_env.sim, 1,
        gymapi.Vec3(0.8, 0.8, 0.8),
        gymapi.Vec3(0.8, 0.8, 0.8),
        gymapi.Vec3(0.5, 0, -1),
    )

    # lidar parameters
    count_min, count_max = 600, 1200
    quality_min, quality_max = 39, 49
    quality_threshold = 40
    raw_lidar_min, raw_lidar_max = 0.2, 12
    lidar_min, lidar_max = 0.2, 10
    rel_lidar_position = options['rel_lidar_position']
    lidar_focus_radius = 4
    lidar_focus_fov = float('inf')

    # input lidar thetas
    thetas = np.linspace(-np.pi, np.pi, options['num_rays'], endpoint=False)

    # control bounds
    qp_control_min = np.array([0, -2])
    qp_control_max = np.array([2, 2])

    # mps parameters
    mps_dt = sim_env.dt
    mps_horizon = 4
    mps_steps = int(mps_horizon/mps_dt)
    num_samples = 1000
    mps_control_min = np.array([0, -2])
    mps_control_max = np.array([2, 2])
    control_center = (mps_control_max + mps_control_min)/2
    noise_scale = (mps_control_max - mps_control_min)/4
    predictive_sampler = PredictiveSampler(
        online_task,
        num_samples,
        mps_control_min,
        mps_control_max,
        noise_scale,
        mps_dt,
    )

    # filter parameters
    filter_threshold = 0.35
    calibration_value_adjustment = -0.5  # from calibrate_value_network.py
    slack_coeff = 1e3

    # safety filter
    safety_filter = SafetyFilter(
        model=model,
        dynamics=dynamics,
        control_min=qp_control_min,
        control_max=qp_control_max,
        filter_threshold=filter_threshold,
        calibration_value_adjustment=calibration_value_adjustment,
        slack_coeff=slack_coeff,
    )

    # disturbance estimation parameters
    prediction_horizon = 2
    prediction_steps = int(prediction_horizon/sim_env.dt)
    window_horizon = 2 # 10 # TODO
    window_steps = int(window_horizon/sim_env.dt)
    coverage = 0.8
    std_width = 2
    dst_dxdy_max_bound = 1
    dst_dth_max_bound = 2

    # reset sim_env and set initial state
    obs = sim_env.reset()
    sim_env.set_robot_state(x=initial_position[0], y=initial_position[1], th=initial_position[2])

    # initialize
    nominal_control_sequence = control_center + np.zeros((mps_steps, 2))
    state = dynamics.wrap_states(sim_env.current_dubins3d_state()[np.newaxis])[0]
    disturbance_estimator = DisturbanceEstimator(
        state,
        prediction_steps,
        window_steps,
        coverage,
        std_width,
        dynamics
    )
    sim_time = 0

    # start SLAM
    if rel_lidar_position[1] != 0:
        print('BreezySLAM cannot handle a LiDAR offset in the y direction')
        raise NotImplementedError
    slam = RMHC_SLAM(LaserModel(offsetMillimeters=-1000*rel_lidar_position[0]), SLAM_MAP_SIZE_PIXELS, SLAM_MAP_SIZE_METERS)
    slam_mapbytes = bytearray(SLAM_MAP_SIZE_PIXELS * SLAM_MAP_SIZE_PIXELS)
    slam_prev_lidar_map = np.zeros((SLAM_MAP_SIZE_PIXELS, SLAM_MAP_SIZE_PIXELS))
    slam_prev_occupancy_map = np.zeros((SLAM_MAP_SIZE_PIXELS, SLAM_MAP_SIZE_PIXELS), dtype=bool)

    # track experiment data
    history = {
        'times': [],
        'v_xs': [],
        'v_yaws': [],
        'nom_v_xs': [],
        'nom_v_yaws': [],
        'raw_angles': [],
        'raw_lidars': [],
        'raw_qualities': [],
        'inp_lidars': [],
        'counts': [],
        'lidar_states': [],
        'states': [],
        'forward_velocities': [],
        'dsts': [],
        'values': [],
        'distance_to_closest_obstacle': [],
        'time_to_goal': None,
        'safe': None,
        'minimum_distance_to_obstacle_set': None,
        'intervention_time_proportion': None,
        'timeout': None,
        'disturbances': [],
        'lidar_focus_radius': lidar_focus_radius,
        'lidar_focus_fov': lidar_focus_fov,
        'prediction_horizon': prediction_horizon,
        'prediction_steps': prediction_steps,
        'window_horizon': window_horizon,
        'window_steps': window_steps,
        'filter_threshold': filter_threshold,
        'calibration_value_adjustment': calibration_value_adjustment,
        'env_path': os.path.join(os.path.basename(ENV_DIR), env_id),
    }

    # run control loop
    intervention_time = 0
    prev_real_time = time.time()
    while True:

        # compute LiDAR position
        lidar_position = state[:2].copy()
        cth, sth = np.cos(state[2]), np.sin(state[2])
        lidar_position[0] = lidar_position[0] + rel_lidar_position[0]*cth - rel_lidar_position[1]*sth
        lidar_position[1] = lidar_position[1] + rel_lidar_position[0]*sth + rel_lidar_position[1]*cth
        
        # compute raw LiDAR
        count = np.random.randint(count_min, count_max)
        raw_angle = np.random.uniform(-np.pi, np.pi, 1400)
        raw_lidar = env.read_lidar(
            lidar_position[np.newaxis],
            raw_angle[np.newaxis]+state[2],
            min_distance=raw_lidar_min,
            max_distance=raw_lidar_max,
        )[0]
        raw_quality = np.random.uniform(quality_min, quality_max, 1400)
        
        # compute input LiDAR
        is_valid = np.logical_and(raw_quality > quality_threshold, np.arange(1400) < count)
        valid_raw_angle, valid_raw_lidar = raw_angle[is_valid], raw_lidar[is_valid]
        sorted_indices = np.argsort(valid_raw_angle)
        sorted_raw_angle, sorted_raw_lidar = valid_raw_angle[sorted_indices], valid_raw_lidar[sorted_indices]
        interp_raw_lidar = interpolate.interp1d(sorted_raw_angle, sorted_raw_lidar, kind='nearest', bounds_error=False, fill_value=(sorted_raw_lidar[0], sorted_raw_lidar[-1]))(thetas)
        inp_lidar = np.minimum(np.maximum(interp_raw_lidar, lidar_min), lidar_max)
        inp_lidar[np.abs(thetas) > lidar_focus_fov] = lidar_max
        inp_lidar_min = np.min(inp_lidar)
        inp_lidar[inp_lidar > inp_lidar_min + lidar_focus_radius] = lidar_max

        # compute slam
        slam_interp_angle_deg = np.arange(360)
        slam_interp_angle_rad = ((slam_interp_angle_deg*np.pi/180)+np.pi)%(2*np.pi)-np.pi
        slam_interp_dist = interpolate.interp1d(sorted_raw_angle, sorted_raw_lidar, kind='nearest', bounds_error=False, fill_value=(sorted_raw_lidar[0], sorted_raw_lidar[-1]))(slam_interp_angle_rad)
        slam.update(list(1000*slam_interp_dist), scan_angles_degrees=list(slam_interp_angle_deg))
        slam.getmap(slam_mapbytes)
        slam_prev_lidar_map = 1 - np.array(slam_mapbytes).reshape(SLAM_MAP_SIZE_PIXELS, SLAM_MAP_SIZE_PIXELS).T[::-1, ::-1]/255
        slam_prev_occupancy_map = np.logical_or(slam_prev_occupancy_map, slam_prev_lidar_map > SLAM_OCCUPANCY_THRESHOLD)
        slam_prev_occupancy_map = np.logical_and(slam_prev_occupancy_map, slam_prev_lidar_map > SLAM_EMPTY_THRESHOLD)
        slam_map = block_reduce(slam_prev_occupancy_map, block_size=SLAM_MAP_SIZE_PIXELS//SLAM_REDUCED_MAP_SIZE_PIXELS, func=np.max)
        slam_environment.set_occupancy_grid(np.logical_or(np.expand_dims(slam_map, axis=-1).astype(bool), np.zeros(slam_environment.occupancy_grid.shape)))

        # compute disturbance
        if len(disturbance_estimator.controls) >= disturbance_estimator.prediction_steps:
            disturbance_bounds = disturbance_estimator.estimate_disturbance_bounds()
            disturbance_norm_bound = np.max(np.abs(disturbance_bounds), axis=0)
            disturbance_bounds_maxnorm = np.array([np.linalg.norm(disturbance_norm_bound[:2]), disturbance_norm_bound[2]])
            dst = np.minimum(disturbance_bounds_maxnorm, np.array([dst_dxdy_max_bound, dst_dth_max_bound])) # cap dst, to conform to input range of NN
        else:
            dst = np.array([dst_dxdy_max_bound, dst_dth_max_bound])/2

        # compute nominal control
        optimal_control_sequence = predictive_sampler.optimal_control_sequence(
            state,
            nominal_control_sequence,
        )
        nom_v_x, nom_v_yaw = optimal_control_sequence[0, 0], optimal_control_sequence[0, 1]

        # compute predicted path
        state_sequence = np.zeros((len(optimal_control_sequence), len(state)))
        for i in range(len(optimal_control_sequence)):
            if i == 0:
                s = state
            else:
                s = state_sequence[i-1]
            state_sequence[i] = dynamics.runge_kutta_step(
                states=s[np.newaxis],
                controls=optimal_control_sequence[np.newaxis, i],
                timesteps=mps_dt,
            )[0]
        state_sequence = dynamics.wrap_states(state_sequence)

        # query network
        if dst is not None:
            pred_value, pred_control = safety_filter.filter(state, inp_lidar, dst, nom_v_x, nom_v_yaw)
        else:
            pred_value, pred_control = None, None

        # extract control
        if pred_control is not None:
            v_x, v_yaw = pred_control
        else:
            v_x, v_yaw = nom_v_x, nom_v_yaw

        # step
        is_filtered = (pred_value is not None) and (pred_value < filter_threshold)
        obs = sim_env.step_dubins3d(v_x, v_yaw, obs, add_line=False, line_color=(0, 0, 0) if is_filtered else (0, 0, 1), body_color=UNSAFE_COLOR if is_filtered else SAFE_COLOR)

        if not HEADLESS:

            # add and refresh drawings
            sim_env.remove_temp_lines()
            
            # draw LiDAR
            lidar_color = (1, 0.5, 0)
            num_lidar_hit_rings = 5
            for i in range(len(thetas)):
                hit_x = lidar_position[0] + inp_lidar[i]*np.cos(thetas[i]+state[2])
                hit_y = lidar_position[1] + inp_lidar[i]*np.sin(thetas[i]+state[2])
                sim_env.add_temp_line(lidar_position[0], lidar_position[1], hit_x, hit_y, lidar_color)
                for r in np.linspace(0.01, 0.05, num_lidar_hit_rings):
                    sim_env.add_temp_box(hit_x, hit_y, 0, r, r, lidar_color)

            # draw predicted path            
            predicted_path_color = (0.5, 0, 1)
            num_predicted_path_points = 10
            num_predicted_path_points_skipped = len(state_sequence)//num_predicted_path_points
            for i in range(0, len(state_sequence)-num_predicted_path_points_skipped, num_predicted_path_points_skipped):
                sim_env.add_temp_line(*state_sequence[i, :2], *state_sequence[i+num_predicted_path_points_skipped, :2], predicted_path_color)

            sim_env.refresh_drawings()

            # plot SLAM
            plt.clf()
            plt.title('SLAM and MPS Planning')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.gca().set_aspect('equal')
            plt.pcolormesh(SLAM_REDUCED_X, SLAM_REDUCED_Y, slam_map.T, cmap='gray', vmin=0, vmax=1)
            state_sequence_in_frame = (state_sequence[:, 0] >= SLAM_REDUCED_X[0])*(state_sequence[:, 0] <= SLAM_REDUCED_X[-1])*(state_sequence[:, 1] >= SLAM_REDUCED_Y[0])*(state_sequence[:, 1] <= SLAM_REDUCED_Y[-1])
            plt.plot(state_sequence[state_sequence_in_frame, 0], state_sequence[state_sequence_in_frame, 1])
            plt.savefig(os.path.join(SAVE_DIR, env_id, f'slam_{str(int(sim_time/sim_env.dt)).zfill(len(str(math.ceil(timeout/sim_env.dt))))}.png'))

        # compute distance to closest obstacle
        inp_lidar_hit_xs = lidar_position[0] + inp_lidar*np.cos(thetas+state[2])
        inp_lidar_hit_ys = lidar_position[1] + inp_lidar*np.sin(thetas+state[2])
        inp_lidar_hit_xys = np.stack((inp_lidar_hit_xs, inp_lidar_hit_ys), axis=-1)
        distance_to_closest_obstacle = np.min(np.linalg.norm(inp_lidar_hit_xys-state[:2], axis=-1))

        # update history
        history['times'].append(sim_time)
        history['v_xs'].append(v_x)
        history['v_yaws'].append(v_yaw)
        history['nom_v_xs'].append(nom_v_x)
        history['nom_v_yaws'].append(nom_v_yaw)
        history['raw_angles'].append(raw_angle)
        history['raw_lidars'].append(raw_lidar)
        history['raw_qualities'].append(raw_quality)
        history['inp_lidars'].append(inp_lidar)
        history['counts'].append(count)
        history['lidar_states'].append(state)
        history['states'].append(state)
        history['forward_velocities'].append(sim_env.base_lin_vel[0, 0].item())
        history['dsts'].append(dst if dst is not None else np.full(2, float('nan')))
        history['values'].append(pred_value if pred_value is not None else float('nan'))
        history['distance_to_closest_obstacle'].append(distance_to_closest_obstacle)

        # set up next iteration
        nominal_control_sequence[:-1] = optimal_control_sequence[1:]
        nominal_control_sequence[-1] = control_center
        state = dynamics.wrap_states(sim_env.current_dubins3d_state()[np.newaxis])[0]
        sim_time += sim_env.dt
        disturbance_estimator.store_observation(np.array([v_x, v_yaw]), sim_env.dt, state)
        if len(disturbance_estimator.controls) >= disturbance_estimator.prediction_steps:
            history['disturbances'].append(disturbance_estimator.get_latest_disturbance())
        else:
            history['disturbances'].append(None)

        if not HEADLESS:
            # save frame
            sim_env.gym.write_viewer_image_to_file(sim_env.viewer, os.path.join(SAVE_DIR, env_id, f'sim_{str(int(sim_time/sim_env.dt)).zfill(len(str(math.ceil(timeout/sim_env.dt))))}.png'))
        
        real_time = time.time()
        print(f'step completed in {real_time-prev_real_time:3.2f} seconds', end='\r')
        prev_real_time = real_time
        
        # update metrics

        if (pred_value is not None) and (pred_value < filter_threshold):
            intervention_time += sim_env.dt
        history['intervention_time_proportion'] = intervention_time/sim_time

        history['minimum_distance_to_obstacle_set'] = min(distance_to_closest_obstacle, history['minimum_distance_to_obstacle_set']) if history['minimum_distance_to_obstacle_set'] is not None else distance_to_closest_obstacle

        in_collision = sim_env.in_collision()
        if in_collision:
            history['safe'] = False
            break

        if task.is_at_goal(state):
            history['time_to_goal'] = sim_time
            history['safe'] = True
            history['timeout'] = False
            break
        
        if sim_time > timeout:
            history['safe'] = True
            history['timeout'] = True
            break

    # close sim
    sim_env.close()

    # save experiment data
    with open(os.path.join(SAVE_DIR, env_id, 'data.pickle'), 'wb') as f:
        pickle.dump(history, f)
    
    if not HEADLESS:
    
        # save videos
        if int(1/sim_env.dt)*sim_env.dt != 1:
            print(f'Framerate will be rounded, since sim_env.dt: {sim_env.dt} does not divide into 1 evenly!')
        
        # save sim video
        os.system(f'ffmpeg -f image2 -r {int(1/sim_env.dt)} -i {os.path.join(SAVE_DIR, env_id, f"sim_%0{len(str(math.ceil(timeout/sim_env.dt)))}d.png")} -vcodec mpeg4 -q:v 1 -y {os.path.join(SAVE_DIR, env_id, "sim_video.mp4")}')

        # save SLAM video
        os.system(f'ffmpeg -f image2 -r {int(1/sim_env.dt)} -i {os.path.join(SAVE_DIR, env_id, f"slam_%0{len(str(math.ceil(timeout/sim_env.dt)))}d.png")} -vcodec mpeg4 -q:v 1 -y {os.path.join(SAVE_DIR, env_id, "slam_video.mp4")}')
        
        # delete temporary frame files
        os.system(f'rm -r {os.path.join(SAVE_DIR, env_id, "*.png")}')

plt.close()