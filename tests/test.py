import time

import gym
import gym_novel_gridworlds

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# env = gym.make('NovelGridworld-v0')
env = gym.make('NovelGridworld-v1')

# Load the trained agent
# model = PPO2.load("ppo_novel_gridworld_v0_100000")
model = PPO2.load("ppo_novel_gridworld_v1_100000")

# env.map_size = 20
# env.items_quantity = {'crafting_table': 2}

for i_episode in range(10):
    print("EPISODE STARTS")
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if i_episode == 0 and i == 0:
            time.sleep(10)
        print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)
        # End the episode if agent is dead
        if done:
            print("Episode #: "+str(i_episode)+" finished after "+str(i)+" timesteps\n")
            time.sleep(1)
            break
