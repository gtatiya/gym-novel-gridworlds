import os
import time

import gym
import gym_novel_gridworlds
import keyboard
import numpy as np
import matplotlib.image as mpimg

from constant import ENV_KEY


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


env_id = 'NovelGridworld-v5'
env = gym.make(env_id)
# env.map_size = 8
obs = env.reset()
env.render()
# fix_item_location('crafting_table', (3, 2))

KEY_ACTION_DICT = ENV_KEY[env_id]


for i in range(100):
    env.render()
    print_play_keys(env.action_str)
    action = get_action_from_keyboard()  # take action from keyboard

    print("action: ", action, env.action_str[action])
    observation, reward, done, info = env.step(action)
    # print("inventory_items_quantity: ", env.inventory_items_quantity)
    print("items_id: ", env.items_id)

    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", len(observation), observation)
    time.sleep(0.2)
    print("")

    if i == 10:
        # env.remap_action()
        # print("action_str: ", env.action_str)
        pass

    if done:
        env.render()
        print("Finished after " + str(i) + " timesteps\n")
        time.sleep(2)
        # env.map_size = np.random.randint(low=10, high=20, size=1)[0]
        observation = env.reset()
        # fix_item_location('crafting_table', (3, 2))

env.close()
