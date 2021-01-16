import os
import time

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.constant import env_key
from gym_novel_gridworlds.wrappers import SaveTrajectories
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import *

import keyboard
import numpy as np
import matplotlib.image as mpimg


def assign_keys(env_id):
    key_action_dict = env_key[env_id]

    action_count = 1
    for action_id in sorted(env.action_str):
        if env.action_str[action_id].startswith('Craft'):
            key_action_dict[str(action_count)] = action_id
            action_count += 1

    alpha_keys = 'abcdefghijklmnopqrstuvwxyz'
    alpha_keys_idx = 0
    for action_id in sorted(env.action_select_str):
        while True:
            if alpha_keys_idx < len(alpha_keys):
                if alpha_keys[alpha_keys_idx] not in key_action_dict:
                    key_action_dict[alpha_keys[alpha_keys_idx]] = action_id
                    alpha_keys_idx += 1
                    break
                else:
                    alpha_keys_idx += 1
            else:
                print("No keys left to assign")
                break

    return key_action_dict

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


env_id = 'NovelGridworld-v6'  # NovelGridworld-v6, NovelGridworld-Bow-v0
env = gym.make(env_id)

# wrappers
env = SaveTrajectories(env, save_path="saved_trajectories")

# observation_wrappers
# env = LidarInFront(env, num_beams=5)
# env = AgentMap(env)

# novelty_wrappers
novelty_name = ''  # axe, axetobreak, fence, additem, replaceitem
# novelty_arg1:
# axe & axetobreak - wooden, iron | fence - oak, jungle | additem - any item name (e.g. paper)
# replaceitem - any existing item (e.g. wall)
novelty_arg1 = 'wall'
# novelty_arg2:
# replaceitem - any item name (e.g. brick)
novelty_arg2 = 'brick'
difficulty = 'medium'  # easy, medium, hard

if novelty_name and novelty_arg1 and difficulty:
    env = inject_novelty(env, difficulty, novelty_name, novelty_arg1, novelty_arg2)

# env = BlockItem(env)
# env = ReplaceItem(env, 'easy', 'wall', 'brick')

if env_id in ['NovelGridworld-v6', 'NovelGridworld-Bow-v0', 'NovelGridworld-Bow-v1']:
    KEY_ACTION_DICT = assign_keys(env_id)
else:
    KEY_ACTION_DICT = env_key[env_id]

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
        # env.block_item(item_to_block='crafting_table', item_to_block_from='tree_log')
        pass

    env.render()
    if done:
        print("Finished after " + str(i) + " timesteps\n")
        time.sleep(2)
        obs = env.reset()
        env.render()

# env.save()
env.close()
