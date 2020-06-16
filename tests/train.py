import gym
import gym_novel_gridworlds

from stable_baselines.common.env_checker import check_env

from stable_baselines import PPO2
from stable_baselines import DQN

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env

env = gym.make('NovelGridworld-v0')
# multiprocess environment
# env = make_vec_env('NovelGridworld-v0', n_envs=4)
check_env(env, warn=True)

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
# model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_novel_world_v0_10000")
