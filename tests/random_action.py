import sys
import time

import gym
import gym_novel_gridworlds
import numpy as np

# sys.path.append('../gym_novel_worlds/envs')
# from novel_world_v0_env import NovelWorldV0Env
# env = NovelWorldV0Env()

env = gym.make('NovelGridworld-v0')
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
print("observation_space.low:", env.observation_space.low)
print("observation_space.high:", env.observation_space.high)
print("sample:", env.observation_space.sample(), env.action_space.sample())

# env.map_size = 7
obs = env.reset()
for i in range(50):
    action_id = env.action_space.sample()  # take a random action
    print("action: ", action_id, list(env.actions_id.keys())[list(env.actions_id.values()).index(action_id)])
    # print("agent_location: ", env.agent_location)
    observation, reward, done, info = env.step(action_id)
    env.render()
    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", observation)
    time.sleep(0)

    if (i+1) % 10 == 0:
        env.map_size = np.random.randint(low=10, high=20, size=1)[0]
        obs = env.reset()
        print("")

env.close()
