import os

import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy


log_dir = 'models'  # "results"
# log_dir = r"C:\Users\GyanT\Documents\GitHub\Reinforcement-Learning\5_DQN\experiments\\"
# agents = ['NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4_lfd', 'NovelGridworld-v5_lfd']
# agents = ['NovelGridworld-v0', 'NovelGridworld-v0_remap_action']
# agents = ['NovelGridworld-v1', 'NovelGridworld-v1_remap_action']
agents = ['NovelGridworld-v0']

plot_after_steps = 1  # 1 for all points

for agent in agents:
    print("agent: ", agent)

    x, y = ts2xy(load_results(log_dir + os.sep + agent), 'timesteps')

    print("# of Episodes: ", len(y))

    # plt.plot(x, y, label=agent)
    plt.plot(x[0::plot_after_steps], y[0::plot_after_steps], label=agent+' ('+str(len(y))+' eps)')

plt.title('Learning Curve')
plt.ylabel("Episodes Rewards")
plt.xlabel("Timesteps")
plt.legend()
plt.savefig(log_dir+os.sep+"learning_curve.png", bbox_inches='tight', dpi=100)
plt.show()
