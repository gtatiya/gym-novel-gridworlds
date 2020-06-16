import time

import gym
import gym_novel_gridworlds
import keyboard
import numpy as np

KEY_ACTION_DICT = {
    "w": 0,  # Forward
    "a": 1,  # Left
    "d": 2,  # Right
}


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


env = gym.make('NovelGridworld-v0')
obs = env.reset()
env.render()

for i in range(100):
    env.render()
    print("Press a key to play: ", KEY_ACTION_DICT)
    action = get_action_from_keyboard()  # take action from keyboard

    print("action: ", action, env.action_str[action])
    observation, reward, done, info = env.step(action)

    print("Step: " + str(i) + ", reward: ", reward)
    # print("observation: ", observation)
    time.sleep(0.2)

    if done:
        env.render()
        print("Finished after " + str(i) + " timesteps\n")
        env.map_size = np.random.randint(low=10, high=20, size=1)[0]
        obs = env.reset()

env.close()
