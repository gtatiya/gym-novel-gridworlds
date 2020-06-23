from gym.envs.registration import register

register(
    id='NovelGridworld-v0',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV0Env',
)

register(
    id='NovelGridworld-v1',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV1Env',
)
