import os
import time

import gym
import gym_novel_gridworlds
from gym_novel_gridworlds.constant import env_key
from gym_novel_gridworlds.wrappers import SaveTrajectories, LimitActions
from gym_novel_gridworlds.observation_wrappers import LidarInFront, AgentMap
from gym_novel_gridworlds.novelty_wrappers import *

import keyboard
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['keymap.quit'].pop(plt.rcParams['keymap.quit'].index('q'))


def assign_keys(env_):

    if hasattr(env_, 'limited_actions_id'):
        actions_id = env_.limited_actions_id
    else:
        actions_id = env_.actions_id

    actions_key = {'Forward': 'w', 'Left': 'a', 'Right': 'd', 'Break': 'e', 'Chop': 'q', 'Jump': 'space',
                   'Place_tree_tap': 'z', 'Extract_rubber': 'x', 'Extract_string': 'x'}

    if env_.env_id in ['NovelGridworld-v6', 'NovelGridworld-Bow-v0', 'NovelGridworld-Bow-v1', 'NovelGridworld-Pogostick-v0', 'NovelGridworld-Pogostick-v1']:
        key_action_id_dict = {}
        for action in actions_key:
            if action in actions_id:
                key_action_id_dict[actions_key[action]] = actions_id[action]

        action_count = 1
        for action in sorted(actions_id):
            if action.startswith('Craft'):
                key_action_id_dict[str(action_count)] = actions_id[action]
                action_count += 1

        alpha_keys = 'abcdefghijklmnopqrstuvwxyz'
        alpha_keys_idx = 0
        for action in sorted(env_.select_actions_id):
            if action not in actions_id:
                continue
            while True:
                if alpha_keys_idx < len(alpha_keys):
                    if alpha_keys[alpha_keys_idx] not in key_action_id_dict:
                        key_action_id_dict[alpha_keys[alpha_keys_idx]] = actions_id[action]
                        alpha_keys_idx += 1
                        break
                    else:
                        alpha_keys_idx += 1
                else:
                    print("No keys left to assign")
                    break
    else:
        key_action_id_dict = env_key[env_id]

    return key_action_id_dict

def print_play_keys(env_, key_action_dict):

    if hasattr(env_, 'limited_actions_id'):
        actions_id = env_.limited_actions_id
    else:
        actions_id = env_.actions_id

    print("Press a key to play: ")
    for key, action_id in key_action_dict.items():
        print(key, ": ", list(actions_id.keys())[list(actions_id.values()).index(action_id)])

def get_action_from_keyboard(key_action_dict):
    while True:
        key_pressed = keyboard.read_key()
        # return index of action if valid key is pressed
        if key_pressed:
            if key_pressed in key_action_dict:
                return key_action_dict[key_pressed]
            elif key_pressed == "esc":
                print("You pressed esc, exiting!!")
                break
            else:
                print("You pressed wrong key. Press Esc key to exit.")

def fix_item_location(item, location):
    result = np.where(env.map == env.items_id[item])
    if len(result) > 0:
        r, c = result[0][0], result[1][0]
        env.map[r][c] = 0
        env.map[location[0]][location[1]] = env.items_id[item]
    else:
        env.map[location[0]][location[1]] = env.items_id[item]


if __name__ == "__main__":
    env_id = 'NovelGridworld-Bow-v0'  # NovelGridworld-v6, NovelGridworld-Bow-v0, NovelGridworld-Pogostick-v0
    env = gym.make(env_id)

    # env.map_size = 12  # np.random.randint(low=10, high=20, size=1)[0]

    # wrappers
    # env = SaveTrajectories(env, save_path="saved_trajectories")
    env = LimitActions(env, {'Forward', 'Left', 'Right', 'Break', 'Craft_bow'})

    # observation_wrappers
    # env = LidarInFront(env, num_beams=8)
    # env = AgentMap(env)

    # novelty_wrappers
    # novelty_name:
    # addchop, additem, addjump, axe, axetobreak, breakincrease, extractincdec, fence, firewall, remapaction, replaceitem
    novelty_name = 'remapaction'
    # novelty_arg1:
    # additem - any item name (e.g. arrow, spring) | axe & axetobreak - iron, wooden |
    # breakincrease - optional: any existing item (e.g. tree_log) | extractincdec - increase or decrease |
    # fence - oak, jungle | replaceitem - any existing item (e.g. wall) |
    novelty_arg1 = ''
    # novelty_arg2:
    # replaceitem - any item name (e.g. brick)
    novelty_arg2 = ''
    # difficulty
    # Only used for: additem, axe, axetobreak, fence, firewall, remapaction, replaceitem
    difficulty = 'medium'  # easy, medium, hard

    if novelty_name:
        env = inject_novelty(env, novelty_name, difficulty, novelty_arg1, novelty_arg2)

    # env = BlockItem(env)
    # env = ReplaceItem(env, 'easy', 'wall', 'brick')

    KEY_ACTION_DICT = assign_keys(env)
    # print("KEY_ACTION_DICT: ", KEY_ACTION_DICT)
    print("action_space:", env.action_space)
    print("actions_id:", len(env.actions_id), env.actions_id)
    print("items_id: ", len(env.items_id), env.items_id)
    print("inventory_items_quantity: ", len(env.inventory_items_quantity), env.inventory_items_quantity)

    # fix_item_location('crafting_table', (3, 2))

    obs = env.reset()
    env.render()
    for i in range(100):
        print_play_keys(env, KEY_ACTION_DICT)
        action_id = get_action_from_keyboard(KEY_ACTION_DICT)  # take action from keyboard
        observation, reward, done, info = env.step(action_id)

        print("action: ", action_id, list(env.actions_id.keys())[list(env.actions_id.values()).index(action_id)])
        print("Step: " + str(i) + ", reward: ", reward)
        print("observation: ", len(observation), observation)

        print("inventory_items_quantity: ", len(env.inventory_items_quantity), env.inventory_items_quantity)

        try:
            print("step_cost, message: ", info['step_cost'], info['message'])
            print("selected_item: ", env.selected_item)
        except:
            pass

        time.sleep(0.2)
        print("")

        if i == 2:
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
