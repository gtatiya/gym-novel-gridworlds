import time

import gym
import gym_novel_gridworlds
import keyboard
import numpy as np

# NovelGridworld-v0
KEY_ACTION_DICT_v0 = {
    "w": 0,  # Forward
    "a": 1,  # Left
    "d": 2,  # Right
}

# NovelGridworld-v1
KEY_ACTION_DICT_v1 = {
    "w": 0,  # Forward
    "a": 1,  # Left
    "d": 2,  # Right
    "e": 3,  # Break
}

KEY_ACTION_DICT = KEY_ACTION_DICT_v1


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
                print("Press a key to play: ", KEY_ACTION_DICT)


env = gym.make('NovelGridworld-v1')
obs = env.reset()
env.render()

for i in range(100):
    env.render()
    print("Press a key to play: ", KEY_ACTION_DICT)
    action = get_action_from_keyboard()  # take action from keyboard

    print("action: ", action, env.action_str[action])
    observation, reward, done, info = env.step(action)
    print("inventory_items_quantity: ", env.inventory_items_quantity)
    print("items_quantity: ", env.items_quantity)

    print("Step: " + str(i) + ", reward: ", reward)
    # print("observation: ", observation)
    time.sleep(0.2)

    if done:
        env.render()
        print("Finished after " + str(i) + " timesteps\n")
        env.map_size = np.random.randint(low=10, high=20, size=1)[0]
        obs = env.reset()

env.close()
