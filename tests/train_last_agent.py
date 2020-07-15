import copy
import os
import time

import gym
import gym_novel_gridworlds

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback


class RenderOnEachStep(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, env):
        super(RenderOnEachStep, self).__init__()
        self.env = env

    def _on_step(self):
        self.env.render()
        # time.sleep(0.5)

"""
This code trains agent an env starting from the terninal state of several agents that acheived terminated state on envs.
Caution: It’s a bad idea to train on intermediate env. because during training episode might not terminate and that will
make next env to not reach goal.
"""

# The agent needs to be trained on last env
env_id_list = ['NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4', 'NovelGridworld-v5', 'NovelGridworld-v4']
# Provide a unique key for each env
env_key_list = ['NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4_tree_tap', 'NovelGridworld-v5',
                'NovelGridworld-v4_pogo_stick']
# Trained agents models except for last env
# env_models = ['ppo_novel_gridworld_v2_lfd_ppo', 'ppo_novel_gridworld_v3_test_train']
# You can also load the model for last env. for training by providing it similarly:
env_models = ['ppo_novel_gridworld_v2_lfd_ppo', 'ppo_novel_gridworld_v3_test_train',
              'ppo_novel_gridworld_v4_lfd_ppo_both', 'ppo_novel_gridworld_v5_lfd_ppo_test_train',
              'ppo_novel_gridworld_v4_lfd_ppo_both']

last_model_name = ''
load_last_model = False
if len(env_id_list) == len(env_models):
    load_last_model = True
else:
    last_model_name = 'ppo_novel_gridworld_v4_test_train'

render = True

log_dir = 'results'

assert len(env_id_list) >= 2, "At least 2 env is needed"

env_dict = {env_key: {} for env_key in env_key_list}
# Load the trained agents
for i in range(len(env_key_list) - 1):
    env_dict[env_key_list[i]]['model'] = PPO2.load(env_models[i])

# make 1st env
env_dict[env_key_list[0]]['env'] = gym.make(env_id_list[0])

for i_episode in range(500):
    # make 2nd env, 3rd env, ... (n-1)th env that can restore previous env
    for i in range(1, len(env_key_list) - 1):
        env_dict[env_key_list[i]]['env'] = gym.make(env_id_list[i], env=env_dict[env_key_list[i - 1]]['env'])

    # make nth env that will be learned
    env_dict[env_key_list[-1]]['env'] = gym.make(env_id_list[-1], env=env_dict[env_key_list[-2]]['env'])
    callback = RenderOnEachStep(env_dict[env_key_list[-1]]['env'])
    log_dir_eps = os.sep.join([log_dir, env_key_list[-1], str(i_episode)])
    os.makedirs(log_dir_eps, exist_ok=True)
    env_dict[env_key_list[-1]]['env'] = Monitor(env_dict[env_key_list[-1]]['env'], log_dir_eps)

    if load_last_model:
        env_dict[env_key_list[-1]]['env'] = DummyVecEnv([lambda: env_dict[env_key_list[-1]]['env']])
        env_dict[env_key_list[-1]]['model'] = PPO2.load(env_models[-1], env_dict[env_key_list[-1]]['env'])
    else:
        if i_episode == 0:
            env_dict[env_key_list[-1]]['model'] = PPO2(MlpPolicy, env_dict[env_key_list[-1]]['env'], verbose=1)
        else:
            env_dict[env_key_list[-1]]['env'] = DummyVecEnv([lambda: env_dict[env_key_list[-1]]['env']])
            env_dict[env_key_list[-1]]['model'] = PPO2.load(last_model_name, env_dict[env_key_list[-1]]['env'])

    print("env_dict: ", env_dict)

    # Play trained env
    for env_idx in range(len(env_key_list) - 1):
        print("EPISODE STARTS: " + env_key_list[env_idx])
        # play each env for 100 steps
        for i in range(100):
            if i == 0:
                obs = env_dict[env_key_list[env_idx]]['env'].reset()  # reset will restore previous env in next env
            action, _states = env_dict[env_key_list[env_idx]]['model'].predict(obs)
            obs, reward, done, info = env_dict[env_key_list[env_idx]]['env'].step(action)
            if render:
                env_dict[env_key_list[env_idx]]['env'].render()
                # time.sleep(0.5)

            if done:
                print("Episode #: " + str(i_episode) + " finished after " + str(i) + " timesteps\n")
                break

    # Learn last env
    if render:
        env_dict[env_key_list[-1]]['model'].learn(total_timesteps=500, callback=callback)
    else:
        env_dict[env_key_list[-1]]['model'].learn(total_timesteps=500)

    if load_last_model:
        env_dict[env_key_list[-1]]['model'].save(env_models[-1])
    else:
        env_dict[env_key_list[-1]]['model'].save(last_model_name)
