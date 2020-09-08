import os
import time

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.constant import env_key
from gym_novel_gridworlds.wrappers import SaveTrajectories
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import Level1Easy, Level1Medium, Level1Hard, BlockItems

import keyboard
import numpy as np
import matplotlib.image as mpimg


def print_play_keys(action_str):
    print("Press a key to play: ")
    for key, key_id in KEY_ACTION_DICT.items():
        print(key, ": ", action_str[key_id])


def get_action_from_keyboard():
    while True:
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
                print_play_keys(env.action_str)


def fix_item_location(item, location):
    result = np.where(env.map == env.items_id[item])
    if len(result) > 0:
        r, c = result[0][0], result[1][0]
        env.map[r][c] = 0
        env.map[location[0]][location[1]] = env.items_id[item]
    else:
        env.map[location[0]][location[1]] = env.items_id[item]


env_id = 'NovelGridworld-v6'
env = gym.make(env_id)
# wrappers
# env = SaveTrajectories(env, save_path="saved_trajectories")

# observation_wrappers
# env = LidarInFront(env)
# env = AgentMap(env)

KEY_ACTION_DICT = env_key[env_id]

# novelty_wrappers
env = BlockItems(env)
level, difficulty = 1, ''  # easy, medium, hard
if level == 1:
    if difficulty == 'easy':
        env = Level1Easy(env)
    elif difficulty == 'medium':
        env = Level1Medium(env)
    elif difficulty == 'hard':
        env = Level1Hard(env)
        KEY_ACTION_DICT.update({"5": len(KEY_ACTION_DICT)})  # Craft_axe

# env.map_size = np.random.randint(low=10, high=20, size=1)[0]
# fix_item_location('crafting_table', (3, 2))

obs = env.reset()
env.render()
for i in range(100):
    print_play_keys(env.action_str)
    action = get_action_from_keyboard()  # take action from keyboard
    observation, reward, done, info = env.step(action)

    print("action: ", action, env.action_str[action])
    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", len(observation), observation)

    print("inventory_items_quantity: ", len(env.inventory_items_quantity), env.inventory_items_quantity)
    print("items_id: ", len(env.items_id), env.items_id)

    try:
        print("step_cost, message: ", info['step_cost'], info['message'])
        print("selected_item: ", env.selected_item)
    except:
        pass

    time.sleep(0.2)
    print("")

    if i == 5:
        # env.remap_action()
        # print("action_str: ", env.action_str)
        # env.add_new_items({'rock': 3, 'axe': 1})
        # env.block_items(item_to_block='crafting_table', item_to_block_from='tree_log')
        pass

    env.render()
    if done:
        print("Finished after " + str(i) + " timesteps\n")
        time.sleep(2)
        obs = env.reset()
        env.render()

# env.save()
env.close()
