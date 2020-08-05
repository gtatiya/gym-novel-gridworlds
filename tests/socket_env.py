import socket
import gym
import gym_novel_gridworlds
import json

env_id = 'NovelGridworld-v0'
env = gym.make(env_id)

# Connect to agent
HOST = '127.0.0.1'
PORT = 9000
sock_agent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_agent.bind((HOST, PORT))
sock_agent.listen()

try:
    conn_agent, addr = sock_agent.accept()
    print('Connected with agent: ', addr)
    env.reset()
    while True:
        action = conn_agent.recv(1024).decode()[:-1]  # get action from agent
        if action == '':
            break
        action_num = list(env.action_str.keys())[list(env.action_str.values()).index(action)]

        obs, reward, done, info = env.step(action_num)
        msg = {'observation': obs.tolist(), 'reward': reward, 'done': done}
        conn_agent.sendall(str.encode(json.dumps(msg) + "\n"))
        env.render()

        print("Action: ", action)
        print("Result: ", msg)
except KeyboardInterrupt:
    pass

sock_agent.shutdown(socket.SHUT_RDWR)
sock_agent.close()
env.close()
