from gym.envs.registration import register

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
