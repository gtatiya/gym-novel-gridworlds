# Author: Gyan Tatiya
# Email: Gyan.Tatiya@tufts.edu

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
    """
    Goal: Break 3 trees
    State: lidar sensor (8 beams) + inventory_items_quantity
    Action: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break'}

    """

    def __init__(self):
        # NovelGridworldV1Env attributes
        self.env_name = 'NovelGridworld-v1'
        self.map_size = 10
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)  # 2D Map
        self.agent_location = (1, 1)  # row, column
        self.direction_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}
        self.agent_facing_str = 'NORTH'
        self.agent_facing_id = self.direction_id[self.agent_facing_str]
        self.block_in_front_str = 'air'
        self.block_in_front_id = 0  # air
        self.block_in_front_location = (0, 0)  # row, column
        self.items = ['crafting_table', 'plank', 'pogo_stick', 'rubber', 'stick', 'tree_log', 'tree_tap', 'wall']
        self.items_id = self.set_items_id(self.items)  # {'crafting_table': 1, 'plank': 2, ...}  # air's ID is 0
        # items_quantity when the episode starts, do not include wall, quantity must be more than 0
        self.items_quantity = {'crafting_table': 1, 'tree_log': 5}
        self.inventory_items_quantity = {item: 0 for item in self.items}
        self.available_locations = []  # locations that do not have item placed
        self.not_available_locations = []  # locations that have item placed or are above, below, left, right to an item

        # Action Space
        self.action_str = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.recipes = {}
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken

        # Observation Space
        self.num_beams = 8
        self.max_beam_range = 40
        self.items_lidar = ['crafting_table', 'tree_log', 'wall']
        self.items_id_lidar = self.set_items_id(self.items_lidar)
        low = np.array([0] * (len(self.items_lidar) * self.num_beams) + [0] * len(self.inventory_items_quantity))
        high = np.array([self.max_beam_range] * (len(self.items_lidar) * self.num_beams) + [40] * len(
            self.inventory_items_quantity))  # maximum 40 stick can be crafted (5 tree_log -> 20 plank -> 40 stick)
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

        # Variables to reset for each reset:
        self.inventory_items_quantity = {item: 0 for item in self.items}
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
        observation = self.get_observation()
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
        Send several beans (self.num_beams) at equally spaced angles in 360 degrees in front of agent within a range
        For each bean store distance (beam_range) for each item in items_id_lidar if item is found otherwise 0
        and return lidar_signals
        """

        direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}

        # Shoot beams in 360 degrees in front of agent
        angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi,
                                  direction_radian[self.agent_facing_str] + np.pi,
                                  self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360

        lidar_signals = []
        r, c = self.agent_location
        for angle in angles_list:
            x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)
            beam_signal = np.zeros(len(self.items_id_lidar), dtype=int)

            # Keep sending longer beams until hit an object or wall
            for beam_range in range(1, self.max_beam_range+1):
                r_obj = r + np.round(beam_range * x_ratio)
                c_obj = c + np.round(beam_range * y_ratio)
                obj_id_rc = self.map[int(r_obj)][int(c_obj)]

                # If bean hit an object or wall
                if obj_id_rc != 0:
                    item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
                    if item in self.items_id_lidar:
                        obj_id_rc = self.items_id_lidar[item]
                        beam_signal[obj_id_rc - 1] = beam_range
                    break

            lidar_signals.extend(beam_signal)

        return lidar_signals

    def set_agent_facing(self, direction_str):

        self.agent_facing_str = direction_str
        self.agent_facing_id = self.direction_id[self.agent_facing_str]

        '''
        self.agent_facing_str = list(self.direction_id.keys())[list(self.direction_id.values()).index(self.agent_facing_id)]
        '''

    def set_items_id(self, items):

        items_id = {}
        for item in sorted(items):
            items_id[item] = len(items_id) + 1

        return items_id

    def get_observation(self):
        """
        observation is lidarSignal + inventory_items_quantity
        :return: observation
        """

        lidar_signals = self.get_lidarSignal()
        observation = lidar_signals + [self.inventory_items_quantity[item] for item in
                                       sorted(self.inventory_items_quantity)]

        return np.array(observation)

    def step(self, action):
        """
        Actions: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break'}
        """

        self.last_action = action
        r, c = self.agent_location

        reward = -1  # default reward
        # Forward
        if action == 0:
            if self.agent_facing_str == 'NORTH' and self.map[r - 1][c] == 0:
                self.agent_location = (r - 1, c)
            elif self.agent_facing_str == 'SOUTH' and self.map[r + 1][c] == 0:
                self.agent_location = (r + 1, c)
            elif self.agent_facing_str == 'WEST' and self.map[r][c - 1] == 0:
                self.agent_location = (r, c - 1)
            elif self.agent_facing_str == 'EAST' and self.map[r][c + 1] == 0:
                self.agent_location = (r, c + 1)
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
        # Break
        elif action == 3:
            self.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if not self.block_in_front_id == 0 and not self.block_in_front_str == 'wall':
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0

                if self.block_in_front_str == 'tree_log':
                    reward = 10
                else:
                    reward = -10  # break something else
                self.inventory_items_quantity[self.block_in_front_str] += 1

        # Update after each step
        observation = self.get_observation()
        self.update_block_in_front()

        done = False
        if self.inventory_items_quantity['tree_log'] >= 3:
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
            self.block_in_front_str = list(self.items_id.keys())[
                list(self.items_id.values()).index(self.block_in_front_id)]

    def render(self, mode='human', title=None):

        color_map = "gist_ncar"

        if title is None:
            title = self.env_name

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

        plt.figure(title, figsize=(9, 5))
        plt.imshow(self.map, cmap=color_map, vmin=0, vmax=len(self.items_id))
        plt.arrow(c, r, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(self.map_size, self.map_size // 2, 'EAST', rotation=90)
        # plt.colorbar()
        # plt.grid()

        info = '\n'.join(["               Info:             ",
                          "Env: "+self.env_name,
                          "Steps: " + str(self.step_count),
                          "Agent Facing: " + self.agent_facing_str,
                          "Action: " + self.action_str[self.last_action],
                          "Reward: " + str(self.last_reward),
                          "Done: " + str(self.last_done)])
        props = dict(boxstyle='round', facecolor='w', alpha=0.2)
        plt.text(-(self.map_size // 2) - 0.5, 2.25, info, fontsize=10, bbox=props)  # x, y

        if self.last_done:
            you_win = "YOU WIN "+self.env_name+"!!!"
            props = dict(boxstyle='round', facecolor='w', alpha=1)
            plt.text(0 - 0.1, (self.map_size // 2), you_win, fontsize=18, bbox=props)

        cmap = get_cmap(color_map)

        legend_elements = [Line2D([0], [0], marker="^", color='w', label='agent', markerfacecolor='w', markersize=12,
                                  markeredgewidth=2, markeredgecolor='k'),
                           Line2D([0], [0], color='w', label="INVENTORY:")]
        for item in sorted(self.inventory_items_quantity):
            rgba = cmap(self.items_id[item] / len(self.items_id))
            legend_elements.append(Line2D([0], [0], marker="s", color='w',
                                          label=item + ': ' + str(self.inventory_items_quantity[item]),
                                          markerfacecolor=rgba, markersize=16))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.55, 1.02))  # x, y

        plt.tight_layout()
        plt.pause(0.01)
        plt.clf()

    def close(self):
        return
