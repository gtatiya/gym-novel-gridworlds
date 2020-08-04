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

ENV_ALGO = {
    'NovelGridworld-v0': PPO2,
    'NovelGridworld-v1': PPO2,
    'NovelGridworld-v2': PPO2,
    'NovelGridworld-v3': PPO2,
    'NovelGridworld-v4': PPO2,
}

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-env", default='NovelGridworld-v0', help="environment ID")
    ap.add_argument("-episodes", default=10, type=int, help="# of episodes")
    args = vars(ap.parse_args())
    # print("args: ", args)

    model_path = os.sep.join(['trained_agents', args['env']])

    if not os.path.exists(model_path+'.zip') and args['env'] in ENV_ALGO.keys():
        print("Model does not exits :(")
        exit()

    env = gym.make(args['env'])

    # Load the trained agent
    if args['env'] == 'NovelGridworld-v5':
        env_id_list = ['NovelGridworld-v1', 'NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4',
                       'NovelGridworld-v3']
        # Provide a unique key for each env
        env_key_list = ['NovelGridworld-v1', 'NovelGridworld-v2', 'NovelGridworld-v3_tree_tap', 'NovelGridworld-v4',
                        'NovelGridworld-v3_pogo_stick']
        render_title = args['env']

        render = True

        env_dict = {env_id: {} for env_id in env_key_list}
        # Load the trained agents
        for i in range(len(env_key_list)):
            model_path = os.sep.join(['trained_agents', env_id_list[i]])
            env_dict[env_key_list[i]]['model'] = ENV_ALGO[env_id_list[i]].load(model_path)

        # make 1st env
        env_dict[env_key_list[0]]['env'] = gym.make(env_id_list[0])

        for i_episode in range(args['episodes']):
            # make 2nd env, 3rd env, ... nth env that can restore previous env
            for i in range(1, len(env_key_list)):
                env_dict[env_key_list[i]]['env'] = gym.make(env_id_list[i], env=env_dict[env_key_list[i - 1]]['env'])

            # Play trained env.
            for env_idx in range(len(env_key_list)):
                print("EPISODE STARTS: " + env_key_list[env_idx])
                # play each env for 100 steps
                # It is possible to not reach goal and move on to next env
                for i in range(100):
                    if i == 0:
                        obs = env_dict[env_key_list[env_idx]][
                            'env'].reset()  # reset will restore previous env in next env
                    action, _states = env_dict[env_key_list[env_idx]]['model'].predict(obs)
                    obs, reward, done, info = env_dict[env_key_list[env_idx]]['env'].step(action)
                    if render:
                        env_dict[env_key_list[env_idx]]['env'].render(title=render_title)
                        # time.sleep(0.5)
                    print("Episode #: " + str(i_episode) + ", step: " + str(i) + ", reward: ", reward)

                    if done:
                        print("Episode #: " + str(i_episode) + " finished after " + str(i) + " timesteps\n")
                        break
    else:
        model = ENV_ALGO[args['env']].load(model_path)

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
