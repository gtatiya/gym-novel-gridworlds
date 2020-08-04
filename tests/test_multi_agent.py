import copy
import os
import time

import gym
import gym_novel_gridworlds

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


# env_id_list = ['NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4',
#                'NovelGridworld-v5', 'NovelGridworld-v4']
env_id_list = ['NovelGridworld-v1', 'NovelGridworld-v2', 'NovelGridworld-v3',
               'NovelGridworld-v4', 'NovelGridworld-v3']
# Provide a unique key for each env
# env_key_list = ['NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4_tree_tap',
#                 'NovelGridworld-v5', 'NovelGridworld-v4_pogo_stick']
env_key_list = ['NovelGridworld-v1', 'NovelGridworld-v2', 'NovelGridworld-v3_tree_tap',
                'NovelGridworld-v4', 'NovelGridworld-v3_pogo_stick']
# env_models = ['NovelGridworld-v2_ppo2_200000', 'NovelGridworld-v3_ppo2_200000', 'NovelGridworld-v4_lfd_ppo2_100000_best_model',
#               'NovelGridworld-v5_lfd_ppo2_200000_last_model', 'NovelGridworld-v4_lfd_ppo2_100000_best_model']
# env_models = ['NovelGridworld-v2_200000_8beams0filled40range3items_in_360degrees_last_model',
#               'NovelGridworld-v3_200000_8beams0filled40range3items_in_360degrees_last_model',
#               'NovelGridworld-v4_200000_8beams0filled40range3items_in_360degrees_lfd_best_model',
#               'NovelGridworld-v5_200000_8beams0filled40range3items_in_360degrees_lfd_best_model',
#               'NovelGridworld-v4_200000_8beams0filled40range3items_in_360degrees_lfd_best_model']
env_models = ['NovelGridworld-v1_200000_8beams0filled40range3items_in_360degrees_last_model',
              'NovelGridworld-v2_200000_8beams0filled40range3items_in_360degrees_last_model',
              'NovelGridworld-v3_200000_8beams0filled40range3items_in_360degrees_lfd_best_model',
              'NovelGridworld-v4_200000_8beams0filled40range3items_in_360degrees_lfd_best_model',
              'NovelGridworld-v3_200000_8beams0filled40range3items_in_360degrees_lfd_best_model']

assert len(env_key_list) == len(env_models), "Provide both: env_id and their models"

render = True

render_title = ''
env_dict = {env_id: {} for env_id in env_key_list}
# Load the trained agents
for i in range(len(env_key_list)):
    print("env_key_list[i]: ", env_key_list[i])
    env_dict[env_key_list[i]]['model'] = PPO2.load(env_models[i])
    render_title += env_key_list[i] + '_'
render_title = render_title[:-1]
render_title = 'NovelGridworld-v5'

# make 1st env
env_dict[env_key_list[0]]['env'] = gym.make(env_id_list[0])

for i_episode in range(10):
    # make 2nd env, 3rd env, ... nth env that can restore previous env
    for i in range(1, len(env_key_list)):
        env_dict[env_key_list[i]]['env'] = gym.make(env_id_list[i], env=env_dict[env_key_list[i - 1]]['env'])

    # Play trained env.
    for env_idx in range(len(env_key_list)):
        print("EPISODE STARTS: " + env_key_list[env_idx])
        # play each env for 100 steps
        # It is possible to not reach goal and move on to next env
        for i in range(100):
            if i == 0:
                obs = env_dict[env_key_list[env_idx]]['env'].reset()  # reset will restore previous env in next env
            action, _states = env_dict[env_key_list[env_idx]]['model'].predict(obs)
            obs, reward, done, info = env_dict[env_key_list[env_idx]]['env'].step(action)
            if render:
                env_dict[env_key_list[env_idx]]['env'].render(title=render_title)
                # time.sleep(0.5)
            print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)

            if done:
                print("Episode #: " + str(i_episode) + " finished after " + str(i) + " timesteps\n")
                break
