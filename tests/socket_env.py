import json
import socket
import time

import gym
import gym_novel_gridworlds


def recv_socket_data(sock):
    BUFF_SIZE = 4096  # 4 KiB
    data = b''
    while True:
        time.sleep(0.00001)
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break

    return data


env_id = 'NovelGridworld-v6'
env = gym.make(env_id)

# Connect to agent
HOST = '127.0.0.1'
PORT = 9000
sock_agent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_agent.bind((HOST, PORT))
sock_agent.listen()
conn_agent, addr = sock_agent.accept()
print('Connected with agent: ', addr)

env.reset()
while True:
    action = recv_socket_data(conn_agent)  # get action from agent
    action = action.decode().strip()
    action_id = env.actions_id[action]

    obs, reward, done, info = env.step(action_id)
    msg = {'observation': str(obs), 'reward': reward, 'done': done}
    conn_agent.sendall(str.encode(json.dumps(msg) + "\n"))
    env.render()

    print("Action: ", action_id, action)
    print("Result: ", msg)

sock_agent.shutdown(socket.SHUT_RDWR)
sock_agent.close()
env.close()
