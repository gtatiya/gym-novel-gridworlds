import time

import gym
import gym_novel_gridworlds
import keyboard

from stable_baselines import DQN

from stable_baselines.gail import generate_expert_traj

from constant import ENV_KEY

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

env_id = 'NovelGridworld-v5'
env = gym.make(env_id)

KEY_ACTION_DICT = ENV_KEY[env_id]


def print_play_keys(action_str):
    print("Press a key to play: ")
    for key, key_id in KEY_ACTION_DICT.items():
        print(key, ": ", action_str[key_id])


def human_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """

    while True:
        env.render()
        print_play_keys(env.action_str)
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
episodes = 50
generate_expert_traj(human_expert, 'expert_' + env_id+'_'+str(episodes)+'demos', env, n_episodes=episodes)
