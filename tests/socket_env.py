import socket

import gym
import gym_novel_gridworlds


env_id = 'NovelGridworld-v0'
env = gym.make(env_id)

# Connect to agent
HOST = '127.0.0.1'
PORT = 9000
sock_agent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_agent.bind((HOST, PORT))
sock_agent.listen()
conn_agent, addr = sock_agent.accept()
print('Connected with agent: ', addr)

obs = env.reset()
for i in range(50):
    action = conn_agent.recv(1024).decode()  # get action from agent
    action = list(env.action_str.keys())[list(env.action_str.values()).index(action)]

    print("action: ", action, env.action_str[action])
    observation, reward, done, info = env.step(action)

    conn_agent.sendall(str.encode(str(observation)))  # send observation to agent

    env.render()
    print("Step: " + str(i) + ", reward: ", reward)
    print("observation: ", observation)

env.close()
