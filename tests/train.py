import os
import time

import gym
import gym_novel_gridworlds

import numpy as np

from stable_baselines.common.env_checker import check_env

from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines.gail import ExpertDataset

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env

from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy

from novelty_generator import remap_action


class RenderOnEachStep(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    """

    def __init__(self, env):
        super(RenderOnEachStep, self).__init__()
        self.env = env

    def _on_step(self):
        self.env.render()
        # time.sleep(0.5)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    """

    def __init__(self, check_freq, log_dir, model_name):
        super(SaveOnBestTrainingRewardCallback, self).__init__()

        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, model_name)
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)


class RemapActionOnEachStep(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    """

    def __init__(self, env, step_num):
        super(RemapActionOnEachStep, self).__init__()
        self.env = env
        self.step_num = step_num

    def _on_step(self):
        if self.n_calls % self.step_num == 0:
            # self.env = remap_action(self.env)
            self.env.remap_action()


if __name__ == "__main__":
    env_id = 'NovelGridworld-v1'
    timesteps = 100000  # 200000
    log_dir = 'models'
    pretrain = False

    env = gym.make(env_id)
    env = Monitor(env, log_dir)
    # callback = RenderOnEachStep(env)
    # callback = SaveOnBestTrainingRewardCallback(1, log_dir, env_id + '_lfd' + '_ppo2_' + str(timesteps) + '_best_model')
    callback = RemapActionOnEachStep(env, 50000)

    # multiprocess environment
    # env = make_vec_env('NovelGridworld-v0', n_envs=4)
    check_env(env, warn=True)

    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1)
    # model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

    # Pretrain the model from human recored dataset
    # specify `traj_limitation=-1` for using the whole dataset
    if pretrain:
        dataset = ExpertDataset(expert_path='expert_NovelGridworld-v5_50demos.npz', traj_limitation=-1, batch_size=128)
        model.pretrain(dataset, n_epochs=2000)

    # model.learn(total_timesteps=timesteps)
    model.learn(total_timesteps=timesteps, callback=callback)

    if pretrain:
        model.save(env_id + '_lfd' + '_ppo2_' + str(timesteps) + '_last_model')
    else:
        # model.save(env_id + '_ppo2_' + str(timesteps) + '_last_model')
        model.save(env_id + '_ppo2_' + str(timesteps) + '_last_model_remap_action')
