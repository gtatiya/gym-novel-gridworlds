# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class NovelGridworldV0Env(gym.Env):
    # metadata = {'render.modes': ['human']}
    """
    Goal: Get in front of the crafting table
    State: lidar sensor (5 beams)
    Action: {0: 'Forward', 1: 'Left', 2: 'Right'}

    """

    def __init__(self):
        # NovelGridworldV0Env attributes
        self.env_name = 'NovelGridworld-v0'
        self.map_size = 10
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)  # 2D Map
        self.agent_location = (1, 1)  # row, column
        self.direction_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}
        self.agent_facing_str = 'NORTH'
        self.agent_facing_id = self.direction_id[self.agent_facing_str]
        self.block_in_front_str = 'air'
        self.block_in_front_id = 0  # air
        self.block_in_front_location = (0, 0)  # row, column
        self.items = ['wall', 'crafting_table']
        self.items_id = self.set_items_id(self.items)  # {'crafting_table': 1, 'wall': 2}  # ID cannot be 0 as air = 0
        self.items_quantity = {'crafting_table': 1}  # Do not include wall, quantity must be more than 0
        self.inventory_items_quantity = {}
        self.available_locations = []  # locations that do not have item placed
        self.not_available_locations = []  # locations that have item placed or are above, below, left, right to an item

        # Action Space
        self.action_str = {0: 'Forward', 1: 'Left', 2: 'Right'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.recipes = {}
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

        observation = self.get_lidarSignal()

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

        # Shoot beams in 180 degrees in front of agent
        angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi / 2,
                                  direction_radian[self.agent_facing_str] + np.pi / 2, self.num_beams)

        lidar_signals = []
        r, c = self.agent_location
        for angle in angles_list:
            x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)

            beam_range = 1
            # beam_signal = np.zeros(len(self.items_id), dtype=int)
            beam_signal = np.full(fill_value=self.max_beam_range, shape=len(self.items_id), dtype=int)

            # Keep sending longer beams until hit an object or wall
            while True:
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
        Actions: {0: 'Forward', 1: 'Left', 2: 'Right'}
        """

        self.last_action = action
        r, c = self.agent_location

        reward = -1  # default reward
        # Forward
        if action == list(self.action_str.keys())[list(self.action_str.values()).index('Forward')]:
            if self.agent_facing_str == 'NORTH' and self.map[r - 1][c] == 0:
                self.agent_location = (r - 1, c)
            elif self.agent_facing_str == 'SOUTH' and self.map[r + 1][c] == 0:
                self.agent_location = (r + 1, c)
            elif self.agent_facing_str == 'WEST' and self.map[r][c - 1] == 0:
                self.agent_location = (r, c - 1)
            elif self.agent_facing_str == 'EAST' and self.map[r][c + 1] == 0:
                self.agent_location = (r, c + 1)
        # Left
        elif action == action == list(self.action_str.keys())[list(self.action_str.values()).index('Left')]:
            if self.agent_facing_str == 'NORTH':
                self.set_agent_facing('WEST')
            elif self.agent_facing_str == 'SOUTH':
                self.set_agent_facing('EAST')
            elif self.agent_facing_str == 'WEST':
                self.set_agent_facing('SOUTH')
            elif self.agent_facing_str == 'EAST':
                self.set_agent_facing('NORTH')
        # Right
        elif action == list(self.action_str.keys())[list(self.action_str.values()).index('Right')]:
            if self.agent_facing_str == 'NORTH':
                self.set_agent_facing('EAST')
            elif self.agent_facing_str == 'SOUTH':
                self.set_agent_facing('WEST')
            elif self.agent_facing_str == 'WEST':
                self.set_agent_facing('NORTH')
            elif self.agent_facing_str == 'EAST':
                self.set_agent_facing('SOUTH')

        # Update after each step
        observation = self.get_lidarSignal()
        self.update_block_in_front()

        done = False
        if self.block_in_front_id == self.items_id['crafting_table']:
            reward = 50
            done = True

        info = {}

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
            self.block_in_front_str = list(self.items_id.keys())[
                list(self.items_id.values()).index(self.block_in_front_id)]

    def remap_action(self):
        """
        Remap actions randomly

        """

        while True:
            actions = list(self.action_str.values())
            random.shuffle(actions)
            action_str_new = dict([(i, action) for i, action in enumerate(actions)])

            if self.action_str != action_str_new:
                self.action_str = action_str_new
                print("New remapped actions: ", self.action_str)
                break

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
        plt.imshow(self.map, cMAP=color_map, vmin=0, vmax=len(self.items_id))
        plt.arrow(c, r, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(self.map_size, self.map_size // 2, 'EAST', rotation=90)
        # plt.colorbar()
        # plt.grid()

        info = '\n'.join(["               Info:             ",
                          "Steps: " + str(self.step_count),
                          "Agent Facing: " + self.agent_facing_str,
                          "Action: " + self.action_str[self.last_action],
                          "Reward: " + str(self.last_reward),
                          "Done: " + str(self.last_done)])
        props = dict(boxstyle='round', facecolor='w', alpha=0.2)
        plt.text(-(self.map_size//2)-0.5, 1.5, info, fontsize=10, bbox=props)

        cmap = get_cmap(color_map)
        legend_elements = [Line2D([0], [0], marker="^", color='w', label='agent', markerfacecolor='w', markersize=12,
                                  markeredgewidth=2, markeredgecolor='k')]
        for item in sorted(self.items_id):
            rgba = cmap(self.items_id[item] / len(self.items_id))
            legend_elements.append(
                Line2D([0], [0], marker="s", color='w', label=item, markerfacecolor=rgba, markersize=16))
        plt.legend(handles=legend_elements, title="Objects:", title_fontsize=12, bbox_to_anchor=(1.5, 1.02))  # x, y

        plt.tight_layout()
        plt.pause(0.01)
        plt.clf()

    def close(self):
        return
