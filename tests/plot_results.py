import os

import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy


log_dir = 'results'
# agents = ['NovelGridworld-v2', 'NovelGridworld-v3', 'NovelGridworld-v4_lfd', 'NovelGridworld-v5_lfd']
# agents = ['NovelGridworld-v0', 'NovelGridworld-v0_remap_action']
# agents = ['NovelGridworld-v2_8beams_in_180degrees', 'NovelGridworld-v2_9beams_in_180degrees',
#           'NovelGridworld-v2_9beams0filled_in_180degrees', 'NovelGridworld-v2_9beams0filled42max_beam_in_180degrees',
#           'NovelGridworld-v2_8beams_in_360degrees']
# agents = ['NovelGridworld-v2_8beams_in_360degrees',
#           'NovelGridworld-v2_8beams0filled40range_in_360degrees',
#           'NovelGridworld-v2_8beams0filled40range3items_in_360degrees']
# agents = ['NovelGridworld-Bow-v0_8beams0filled11hypotenuserange3items_in_360degrees']
# agents = ['NovelGridworld-Bow-v0_A2C']
# agents = ['NovelGridworld-Bow-v0_8beams0filled11hypotenuserange3items_in_360degrees']
agents = ['NovelGridworld-Bow-v0_']
# log_dir = r"C:\Users\GyanT\Documents\GitHub\Reinforcement-Learning\5_DQN\experiments\\"
# agents = ['NovelGridworld-v0_1_DQN', 'NovelGridworld-v0_2_Dueling_DQN', 'NovelGridworld-v0_3_Dueling_Double_DQN',
#           'NovelGridworld-v0_4_Double_PER_DQN', 'NovelGridworld-v0_5_Dueling_Double_PER_DQN']
# agents = ['NovelGridworld-v0_3.1_Double_PER_DQN']

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
plt.savefig(log_dir + os.sep + "learning_curve.png", bbox_inches='tight', dpi=600)
plt.show()
