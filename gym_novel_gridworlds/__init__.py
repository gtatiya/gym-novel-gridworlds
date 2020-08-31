from gym.envs.registration import register
import gym_novel_gridworlds.constant
import gym_novel_gridworlds.wrappers
import gym_novel_gridworlds.novelty_wrappers
import gym_novel_gridworlds.observation_wrappers

register(
    id='NovelGridworld-v0',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV0Env',
)

register(
    id='NovelGridworld-v1',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV1Env',
)

register(
    id='NovelGridworld-v2',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV2Env',
)

register(
    id='NovelGridworld-v3',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV3Env',
)

register(
    id='NovelGridworld-v4',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV4Env',
)

register(
    id='NovelGridworld-v5',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV5Env',
)

register(
    id='NovelGridworld-v6',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV6Env',
)
