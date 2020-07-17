import socket

import gym
import gym_novel_gridworlds


# Connect to NovelGridworld
HOST = '127.0.0.1'
PORT = 9000
sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_game.connect((HOST, PORT))

env_id = 'NovelGridworld-v0'
env = gym.make(env_id)

while True:
    action = env.action_space.sample()  # random action
    print("Sending action: ", action, env.action_str[action])
    sock_game.send(str.encode(env.action_str[action]))  # send action to env

    observation = sock_game.recv(1024).decode()  # get observation from env
    print("Received observation: ", observation)
