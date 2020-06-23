# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class NovelGridworldV1Env(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # NovelGridworldV1Env attributes
        self.env_name = 'NovelGridworld-v1'
        self.map_size = 10
        self.agent_location = (1, 1)  # row, column
        self.direction_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}
        self.agent_facing_str = 'NORTH'
        self.agent_facing_id = self.direction_id[self.agent_facing_str]
        self.block_in_front_str = 'air'
        self.block_in_front_id = 0  # Air
        self.block_in_front_location = (0, 0)  # row, column
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)  # 2D Map
        self.items = ['wall', 'crafting_table', 'tree']
        self.items_id = self.set_items_id(self.items)  # {'crafting_table': 1, 'wall': 2}  # ID cannot be 0 as air = 0
        self.items_quantity = {'crafting_table': 1, 'tree': 5}  # Do not include wall, quantity must be more than  0
        self.inventory_items_quantity = {}
        self.available_locations = []  # locations that do not have item placed
        self.not_available_locations = []  # locations that have item placed or are above, below, left, right to an item

        # Action Space
        # 0=Forward, 1=Left, 2=Right
        self.action_str = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken

        # Observation Space
        self.num_beams = 5
        self.max_beam_range = int(math.sqrt(2 * (self.map_size - 2) ** 2))  # Hypotenuse of a square
        low = np.ones(len(self.items_id) * self.num_beams, dtype=int)
        high = np.array([self.max_beam_range] * len(self.items_id) * self.num_beams)
        self.observation_space = spaces.Box(low, high, dtype=int)

        # Reward
        self.last_reward = 0  # last received reward

        self.last_done = False  # last done

    def reset(self, map_size=None, items_id=None, items_quantity=None):

        if map_size is not None:
            self.map_size = map_size
        if items_id is not None:
            self.items_id = items_id
        if items_quantity is not None:
            self.items_quantity = items_quantity

        # Assertions and assumptions
        assert len(self.items_id) == len(self.items_quantity) + 1, "Should be equal, otherwise color might be wrong"

        # Variables to reset for each reset:
        self.inventory_items_quantity = {}
        self.available_locations = []
        self.not_available_locations = []
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken
        self.last_reward = 0  # last received reward
        self.last_done = False  # last done

        self.map = np.zeros((self.map_size - 2, self.map_size - 2), dtype=int)  # air=0
        self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=self.items_id['wall'])

        """
        available_locations: locations 1 block away from the wall are valid locations to place items and agent
        available_locations: locations that do not have item placed
        """
        for r in range(2, self.map_size - 2):
            for c in range(2, self.map_size - 2):
                self.available_locations.append((r, c))

        # Agent
        idx = np.random.choice(len(self.available_locations), size=1)[0]
        self.agent_location = self.available_locations[idx]

        # Agent facing direction
        self.set_agent_facing(direction_str=np.random.choice(list(self.direction_id.keys()), size=1)[0])

        for item, quantity in self.items_quantity.items():
            self.add_item_to_map(item, num_items=quantity)

        if self.agent_location not in self.available_locations:
            self.available_locations.append(self.agent_location)

        # Update after each reset
        observation = self.get_lidarSignal()
        self.update_block_in_front()

        return observation

    def add_item_to_map(self, item, num_items):

        item_id = self.items_id[item]

        count = 0
        while True:
            if num_items == count:
                break
            assert not len(self.available_locations) < 1, "Cannot place items, increase map size!"

            idx = np.random.choice(len(self.available_locations), size=1)[0]
            r, c = self.available_locations[idx]

            if (r, c) == self.agent_location:
                self.available_locations.pop(idx)
                continue

            # If at (r, c) is air, and its North, South, West and East are also air, add item
            if (self.map[r][c]) == 0 and (self.map[r - 1][c] == 0) and (self.map[r + 1][c] == 0) and (
                    self.map[r][c - 1] == 0) and (self.map[r][c + 1] == 0):
                self.map[r][c] = item_id
                count += 1
            self.not_available_locations.append(self.available_locations.pop(idx))

    def get_lidarSignal(self):
        """
        Send several beans (self.num_beams) at equally spaced angles in front of agent
        For each bean store distance (beam_range) for each item if item is found otherwise self.max_beam_range
        and return lidar_signals
        """
        direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}

        # All directions
        angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi / 2,
                                  direction_radian[self.agent_facing_str] + np.pi / 2, self.num_beams)

        lidar_signals = []
        for angle in angles_list:
            x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)

            beam_range = 1
            # beam_signal = np.zeros(len(self.items_id), dtype=int)
            beam_signal = np.full(fill_value=self.max_beam_range, shape=len(self.items_id), dtype=int)

            # Keep sending longer beams until hit an object or wall
            while True:
                r, c = self.agent_location
                r_obj = r + np.round(beam_range * x_ratio)
                c_obj = c + np.round(beam_range * y_ratio)
                obj_id_rc = self.map[int(r_obj)][int(c_obj)]

                # If bean hit an object or wall
                if obj_id_rc != 0:
                    beam_signal[obj_id_rc - 1] = beam_range
                    break

                beam_range += 1

            lidar_signals.extend(beam_signal)

        return np.array(lidar_signals)

    def set_agent_facing(self, direction_str):

        self.agent_facing_str = direction_str
        self.agent_facing_id = self.direction_id[self.agent_facing_str]

        '''
        self.agent_facing_str = list(self.direction_id.keys())[list(self.direction_id.values()).index(self.agent_facing_id)]
        '''

    def set_items_id(self, items):

        self.items_id = {}
        for item in sorted(items):
            self.items_id[item] = len(self.items_id) + 1

        return self.items_id

    def step(self, action):
        """
        Actions: 0=Forward, 1=Left, 2=Right
        """
        self.last_action = action
        r, c = self.agent_location
        old_agent_location = copy.deepcopy(self.agent_location)
        old_agent_facing_str = copy.deepcopy(self.agent_facing_str)

        reward = -1  # If agent do not move/turn
        # Forward
        if action == 0:
            if self.agent_facing_str == 'NORTH' and self.map[r - 1][c] == 0:
                self.agent_location = [r - 1, c]
            elif self.agent_facing_str == 'SOUTH' and self.map[r + 1][c] == 0:
                self.agent_location = [r + 1, c]
            elif self.agent_facing_str == 'WEST' and self.map[r][c - 1] == 0:
                self.agent_location = [r, c - 1]
            elif self.agent_facing_str == 'EAST' and self.map[r][c + 1] == 0:
                self.agent_location = [r, c + 1]

            # if old_agent_location != self.agent_location:
            #     reward = -1  # If agent moved
        # Left
        elif action == 1:
            if self.agent_facing_str == 'NORTH':
                self.set_agent_facing('WEST')
            elif self.agent_facing_str == 'SOUTH':
                self.set_agent_facing('EAST')
            elif self.agent_facing_str == 'WEST':
                self.set_agent_facing('SOUTH')
            elif self.agent_facing_str == 'EAST':
                self.set_agent_facing('NORTH')

            # if old_agent_facing_str != self.agent_facing_str:
            #     reward = -1  # If agent turned
        # Right
        elif action == 2:
            if self.agent_facing_str == 'NORTH':
                self.set_agent_facing('EAST')
            elif self.agent_facing_str == 'SOUTH':
                self.set_agent_facing('WEST')
            elif self.agent_facing_str == 'WEST':
                self.set_agent_facing('NORTH')
            elif self.agent_facing_str == 'EAST':
                self.set_agent_facing('SOUTH')

            # if old_agent_facing_str != self.agent_facing_str:
            #     reward = -1  # If agent turned
        # Break
        elif action == 3:
            self.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if not self.block_in_front_id == 0 and not self.block_in_front_str == 'wall':
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0

                self.inventory_items_quantity.setdefault(self.block_in_front_str, 0)
                self.inventory_items_quantity[self.block_in_front_str] += 1

                if self.block_in_front_str == 'tree':
                    reward = 10

        # Update after each step
        observation = self.get_lidarSignal()
        self.update_block_in_front()

        done = False
        if 'tree' in self.inventory_items_quantity and self.inventory_items_quantity['tree'] >= 3:
            reward = 50
            done = True

        info = {}

        # Update after each step
        self.step_count += 1
        self.last_reward = reward
        self.last_done = done

        return observation, reward, done, info

    def update_block_in_front(self):
        r, c = self.agent_location

        if self.agent_facing_str == 'NORTH':
            self.block_in_front_id = self.map[r - 1][c]
            self.block_in_front_location = (r - 1, c)
        elif self.agent_facing_str == 'SOUTH':
            self.block_in_front_id = self.map[r + 1][c]
            self.block_in_front_location = (r + 1, c)
        elif self.agent_facing_str == 'WEST':
            self.block_in_front_id = self.map[r][c - 1]
            self.block_in_front_location = (r, c - 1)
        elif self.agent_facing_str == 'EAST':
            self.block_in_front_id = self.map[r][c + 1]
            self.block_in_front_location = (r, c + 1)

        if self.block_in_front_id == 0:
            self.block_in_front_str = 'air'
        else:
            self.block_in_front_str = list(self.items_id.keys())[list(self.items_id.values()).index(self.block_in_front_id)]

    def render(self, mode='human'):

        color_map = "gist_ncar"

        r, c = self.agent_location
        x2, y2 = 0, 0
        if self.agent_facing_str == 'NORTH':
            x2, y2 = 0, -0.01
        elif self.agent_facing_str == 'SOUTH':
            x2, y2 = 0, 0.01
        elif self.agent_facing_str == 'WEST':
            x2, y2 = -0.01, 0
        elif self.agent_facing_str == 'EAST':
            x2, y2 = 0.01, 0

        plt.figure(self.env_name, figsize=(9, 5))
        plt.imshow(self.map, cMAP=color_map)
        plt.arrow(c, r, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(self.map_size, self.map_size // 2, 'EAST', rotation=90)
        # plt.colorbar()
        # plt.grid()

        # legend_elements = [Line2D([0], [0], color='w', label="Agent Facing: " + self.agent_facing_str),
        #                    Line2D([0], [0], color='w', label="Action: " + self.action_str[self.last_action]),
        #                    Line2D([0], [0], color='w', label="Reward: " + str(self.last_reward))]
        # legend1 = plt.legend(handles=legend_elements, title="Info:", title_fontsize=12,
        #                      bbox_to_anchor=(1.62, 0.7))  # x, y

        info = '\n'.join(["               Info:             ",
                          "Steps: " + str(self.step_count),
                          "Agent Facing: " + self.agent_facing_str,
                          "Action: " + self.action_str[self.last_action],
                          "Reward: " + str(self.last_reward),
                          "Done: " + str(self.last_done)])
        props = dict(boxstyle='round', facecolor='w', alpha=0.2)
        plt.text(-(self.map_size//2)-0.5, 1.5, info, fontsize=10, bbox=props)

        cmap = get_cmap(color_map)

        legend_elements = [Line2D([0], [0], color='w', label="ITEMS:"),
                           Line2D([0], [0], marker="^", color='w', label='agent', markerfacecolor='w', markersize=12,
                                  markeredgewidth=2, markeredgecolor='k')]
        for item in sorted(self.items_id):
            rgba = cmap(self.items_id[item] / len(self.items_id))
            legend_elements.append(
                Line2D([0], [0], marker="s", color='w', label=item, markerfacecolor=rgba, markersize=16))

        if len(self.inventory_items_quantity) == 0:
            legend_elements.append(Line2D([0], [0], color='w', label="empty INVENTORY"))
        else:
            legend_elements.append(Line2D([0], [0], color='w', label="INVENTORY:"))
            for item in sorted(self.inventory_items_quantity):
                rgba = cmap(self.items_id[item] / len(self.items_id))
                legend_elements.append(Line2D([0], [0], marker="s", color='w',
                                              label=item+': '+str(self.inventory_items_quantity[item]),
                                              markerfacecolor=rgba, markersize=16))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.5, 1.02))  # x, y
        # plt.gca().add_artist(legend1)

        plt.tight_layout()
        plt.pause(0.01)
        plt.clf()

    def close(self):
        return
