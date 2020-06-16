# Gym Novel Gridworlds

Gym Novel Gridworlds are environments for [OpenAI Gym](https://github.com/openai/gym).

## Installation
```
git clone https://github.com/gtatiya/gym-novel-gridworlds.git
cd gym-novel-gridworlds
pip install -e .
```

## Environments

### NovelGridworld

<img src="pics/NovelGridworld-v0.gif" alt="drawing" width="800"/>

## Running
```
import gym
import gym_novel_gridworlds

env = gym.make('NovelGridworld-v0')

done = False
while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

env.close()
```
