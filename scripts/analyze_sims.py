import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

ENV_DIR = 'data/environments/simulation'
SAVE_DIR = 'results/simulation'

env_ids = []
safes = []
minimum_distances_to_obstacle_set = []
intervention_time_proportions = []
timeouts = []
total_distances = []
total_times = []

for env_id in os.listdir(SAVE_DIR):
    if not os.path.exists(os.path.join(SAVE_DIR, env_id, 'data.pickle')):
        continue
    with open(os.path.join(ENV_DIR, env_id, 'environment.pickle'), 'rb') as f:
        env = pickle.load(f)
    with open(os.path.join(SAVE_DIR, env_id, 'data.pickle'), 'rb') as f:
        data = pickle.load(f)
    env_ids.append(env_id)
    safes.append(data['safe'])
    minimum_distances_to_obstacle_set.append(data['minimum_distance_to_obstacle_set'])
    intervention_time_proportions.append(data['intervention_time_proportion'])
    timeouts.append(data['timeout'])
    total_distance = 0
    prev_state = None
    for state in data['states']:
        if prev_state is not None:
            total_distance += np.linalg.norm(np.asarray(state[:2])-np.asarray(prev_state[:2]))
        prev_state = state
    total_distances.append(total_distance)
    total_times.append(data['times'][-1]-data['times'][0])
if len(env_ids) == 0:
    print(f'No runs found in {SAVE_DIR}.')
    quit()
env_ids = np.array(env_ids)
safes = np.array(safes).astype(bool)
minimum_distances_to_obstacle_set = np.array(minimum_distances_to_obstacle_set)
intervention_time_proportions = np.array(intervention_time_proportions)
timeouts = np.array(timeouts).astype(bool)
total_distances = np.array(total_distances)
total_times = np.array(total_times)

safe_completions = np.logical_and(safes, np.logical_not(timeouts)).astype(bool)

print(f'num runs: {len(env_ids)}')
print(f'safe completion rate: {np.mean(safe_completions)}')
print(f'collision rate: {np.mean(np.logical_not(safes))}')
print(f'timeout rate: {np.mean(timeouts)}')
print(f'average minimum distance to obstacle set on safely completed runs: {np.mean(minimum_distances_to_obstacle_set[safe_completions])}')
print(f'average intervention time proportion on safely completed runs: {np.mean(intervention_time_proportions[safe_completions])}')
print(f'average velocity on safely completed runs: {np.mean(total_distances[safe_completions]/total_times[safe_completions])}')