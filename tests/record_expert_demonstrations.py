import time

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap

import keyboard

from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

from keyboard_interface import assign_keys, print_play_keys

"""
Generate Expert Trajectories from a model
"""

# env_id = 'NovelGridworld-v2'
# model = DQN('MlpPolicy', env_id, verbose=1)
#
# # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
# # data will be saved in a numpy archive named `expert_+env_id.npz`
# generate_expert_traj(model, 'expert_'+env_id, n_timesteps=int(10), n_episodes=5)

"""
Generate Expert Trajectories from a human expert player
"""

env_id = 'NovelGridworld-Bow-v0'
env = gym.make(env_id)
env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'})
env = LidarInFront(env)

KEY_ACTION_DICT = assign_keys(env)


def human_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """

    while True:
        env.render()
        print_play_keys(env, KEY_ACTION_DICT)
        time.sleep(0.2)
        key_pressed = keyboard.read_key()
        # return index of action if valid key is pressed
        if key_pressed:
            if key_pressed in KEY_ACTION_DICT:
                return KEY_ACTION_DICT[key_pressed]
            elif key_pressed == "esc":
                print("You pressed esc, exiting!!")
                break
            else:
                print("You pressed wrong key. Press Esc key to exit, OR:")


# Data will be saved in a numpy archive named `expert_+env_id.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly
env.render()
episodes = 10
generate_expert_traj(human_expert, 'expert_' + env_id + '_' + str(episodes) + 'demos', env, n_episodes=episodes)
