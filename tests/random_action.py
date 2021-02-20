import sys
import time

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.novelty_wrappers import inject_novelty
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions

import numpy as np

# sys.path.append('../gym_novel_worlds/envs')
# from novel_world_v0_env import NovelWorldV0Env
# env = NovelWorldV0Env()

env_id = 'NovelGridworld-Bow-v0'  # NovelGridworld-v6, NovelGridworld-Bow-v0, NovelGridworld-Pogostick-v0
env = gym.make(env_id)

# wrappers
# env = SaveTrajectories(env, save_path="saved_trajectories")
env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'})

# observation_wrappers
env = LidarInFront(env, num_beams=8)
# env = AgentMap(env)

# novelty_wrappers
# novelty_name:
# addchop, additem, axe, axetobreak, breakincrease, extractincdec, fence, firewall, remapaction, replaceitem
novelty_name = 'breakincrease'
# novelty_arg1:
# additem - any item name (e.g. arrow, spring) | axe & axetobreak - iron, wooden |
# breakincrease - optional: any existing item (e.g. tree_log) | extractincdec - increase or decrease |
# fence - oak, jungle | replaceitem - any existing item (e.g. wall) |
novelty_arg1 = ''
# novelty_arg2:
# replaceitem - any item name (e.g. brick)
novelty_arg2 = ''
difficulty = 'medium'  # easy, medium, hard

if novelty_name:
    env = inject_novelty(env, novelty_name, difficulty, novelty_arg1, novelty_arg2)

print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
# print("observation_space.low:", env.observation_space.low)
# print("observation_space.high:", env.observation_space.high)
print("sample:", env.observation_space.sample(), env.action_space.sample())

# env.map_size = 7
obs = env.reset()
for i in range(50):
    action_id = env.action_space.sample()  # take a random action
    print("action: ", action_id, list(env.actions_id.keys())[list(env.actions_id.values()).index(action_id)])
    # print("agent_location: ", env.agent_location)
    obs, reward, done, info = env.step(action_id)
    env.render()
    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", obs)
    time.sleep(0)

    if (i+1) % 10 == 0:
        env.map_size = np.random.randint(low=10, high=20, size=1)[0]
        obs = env.reset()
        print("")

env.close()
