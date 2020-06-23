import argparse
import os
import time

import gym
import gym_novel_gridworlds
try:
    import mpi4py
except ImportError:
    mpi4py = None

from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3
# DDPG and TRPO require MPI to be installed
if mpi4py is None:
    DDPG, TRPO = None, None
else:
    from stable_baselines import DDPG, TRPO


ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-env", default='NovelGridworld-v0', help="environment ID")
    ap.add_argument("-algo", default='ppo2', help="RL Algorithm")
    ap.add_argument("-episodes", default=10, type=int, help="# of episodes")
    args = vars(ap.parse_args())
    # print("args: ", args)

    model_path = os.sep.join(['trained_agents', args['algo'], args['env']])

    if not os.path.exists(model_path+'.zip'):
        print("Model does not exits :(")
        exit()

    env = gym.make(args['env'])

    # Load the trained agent
    model = ALGOS[args['algo']].load(model_path)

    for i_episode in range(args['episodes']):
        print("EPISODE STARTS")
        obs = env.reset()
        for i in range(100):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)
            # End the episode if agent is dead
            if done:
                print("Episode #: " + str(i_episode) + " finished after " + str(i) + " timesteps\n")
                time.sleep(1)
                break
