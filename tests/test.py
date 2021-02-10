import time

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env_id = 'NovelGridworld-Bow-v0'  # 'NovelGridworld-v1'
env = gym.make(env_id)
env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'})
env = LidarInFront(env)

# Load the trained agent
model = PPO2.load('NovelGridworld-Bow-v0_400000_8beams0filled11hypotenuserange3items_in_360degrees_best_model')

# env.map_size = 20
# env.items_quantity = {'crafting_table': 2}
# env.action_str = {0: 'Forward', 1: 'Right', 2: 'Left'}

for i_episode in range(10):
    print("EPISODE STARTS")
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        # if i_episode == 0 and i == 0:
        #     time.sleep(10)
        print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)
        # End the episode if agent is dead
        if done:
            print("Episode #: "+str(i_episode)+" finished after "+str(i)+" timesteps\n")
            time.sleep(1)
            break
