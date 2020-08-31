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


class NovelGridworldV2Env(gym.Env):
    # metadata = {'render.modes': ['human']}
    """
    Goal: Craft 8 plank and 8 stick
    State: lidar sensor (8 beams) + inventory_items_quantity
    Action: {0: 'Craft_plank', 1: 'Craft_stick'}

    """

    def __init__(self, env=None):
        # NovelGridworldV2Env attributes
        self.env_name = 'NovelGridworld-v2'
        self.env = env  # env to restore in reset
        self.map_size = 10
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)  # 2D Map
        self.agent_location = (1, 1)  # row, column
        self.direction_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}
        self.agent_facing_str = 'NORTH'
        self.agent_facing_id = self.direction_id[self.agent_facing_str]
        self.block_in_front_str = 'air'
        self.block_in_front_id = 0  # air
        self.block_in_front_location = (0, 0)  # row, column
        self.items = ['wall', 'crafting_table', 'tree_log', 'pogo_stick', 'stick', 'plank', 'rubber', 'tree_tap']
        self.items_id = self.set_items_id(self.items)  # {'crafting_table': 1, 'plank': 2, ...}  # air's ID is 0
        # items_quantity when the episode starts, do not include wall, quantity must be more than 0
        self.items_quantity = {'crafting_table': 1, 'tree_log': 2}
        self.inventory_items_quantity = {item: 0 for item in self.items}
        self.inventory_items_quantity['tree_log'] = 3  # to enable crafting 8 plank and 8 stick directly
        self.available_locations = []  # locations that do not have item placed
        self.not_available_locations = []  # locations that have item placed or are above, below, left, right to an item

        # Action Space
        self.action_str = {0: 'Craft_plank', 1: 'Craft_stick'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.recipes = {'pogo_stick': {'input': {'stick': 4, 'plank': 2, 'rubber': 1}, 'output': {'pogo_stick': 1}},
                        'stick': {'input': {'plank': 2}, 'output': {'stick': 4}},
                        'plank': {'input': {'tree_log': 1}, 'output': {'plank': 4}},
                        'tree_tap': {'input': {'plank': 5, 'stick': 1}, 'output': {'tree_tap': 1}},
                        'crafting_table': {'input': {'plank': 4}, 'output': {'crafting_table': 1}}}
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken

        # Observation Space
        self.num_beams = 8
        self.max_beam_range = 40
        self.items_lidar = ['wall', 'crafting_table', 'tree_log']
        self.items_id_lidar = self.set_items_id(self.items_lidar)
        low = np.array([0] * (len(self.items_lidar) * self.num_beams) + [0] * len(self.inventory_items_quantity))
        high = np.array([self.max_beam_range] * (len(self.items_lidar) * self.num_beams) + [40] * len(
            self.inventory_items_quantity))  # maximum 40 stick can be crafted (5 tree_log -> 20 plank -> 40 stick)
        self.observation_space = spaces.Box(low, high, dtype=int)

        # Reward
        self.last_reward = 0  # last received reward

        self.last_done = False  # last done

    def reset(self, map_size=None, items_id=None, items_quantity=None):

        print("RESETTING " + self.env_name + " ...")
        if self.env is not None:
            print("RESTORING "+self.env_name+" ...")
            self.map_size = copy.deepcopy(self.env.map_size)
            self.map = copy.deepcopy(self.env.map)
            self.items_id = copy.deepcopy(self.env.items_id)
            self.items_quantity = copy.deepcopy(self.env.items_quantity)
            self.inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)
            self.available_locations = copy.deepcopy(self.env.available_locations)
            self.not_available_locations = copy.deepcopy(self.env.not_available_locations)
            self.last_action = copy.deepcopy(self.env.last_action)  # last actions executed
            self.step_count = copy.deepcopy(self.env.step_count)  # no. of steps taken
            self.last_reward = copy.deepcopy(self.env.last_reward)  # last received reward
            self.last_done = False  # last done
            self.agent_location = copy.deepcopy(self.env.agent_location)
            self.agent_facing_str = copy.deepcopy(self.env.agent_facing_str)
            self.agent_facing_id = copy.deepcopy(self.env.agent_facing_id)

            observation = self.get_observation()
            self.update_block_in_front()

            return observation

        if map_size is not None:
            self.map_size = map_size
        if items_id is not None:
            self.items_id = items_id
        if items_quantity is not None:
            self.items_quantity = items_quantity

        # Variables to reset for each reset:
        self.inventory_items_quantity = {item: 0 for item in self.items}
        self.inventory_items_quantity['tree_log'] = 3  # to enable crafting 8 plank and 8 stick directly
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
        Actions: {0: 'Craft_plank', 1: 'Craft_stick'}
        """

        self.last_action = action
        r, c = self.agent_location

        reward = -1  # default reward
        # Craft_plank
        if action == 0:
            item_to_craft = 'plank'
            reward = self.craft(item_to_craft)
        # Craft_stick
        elif action == 1:
            item_to_craft = 'stick'
            reward = self.craft(item_to_craft)

        # Update after each step
        observation = self.get_observation()
        self.update_block_in_front()

        done = False
        if self.inventory_items_quantity['plank'] >= 8 and self.inventory_items_quantity['stick'] >= 8:
            reward = 50
            done = True
        elif not self.has_ingredients_to_craft('plank') and not self.has_ingredients_to_craft('stick'):
            print("Sorry you can't craft anything")
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

    def craft(self, item_to_craft):

        reward = -1  # default reward to craft in a wrong way

        have_all_ingredients = {}
        for item in self.recipes[item_to_craft]['input']:
            if self.inventory_items_quantity[item] >= self.recipes[item_to_craft]['input'][item]:
                have_all_ingredients[item] = True
            else:
                have_all_ingredients[item] = False

        if False in have_all_ingredients.values():
            # print("You don't have:")
            for item in have_all_ingredients:
                if not have_all_ingredients[item]:
                    # print(str(self.recipes[item_to_craft]['input'][item]) + ' ' + item)
                    pass
        else:
            for item in self.recipes[item_to_craft]['input']:
                self.inventory_items_quantity[item] -= self.recipes[item_to_craft]['input'][item]
            self.inventory_items_quantity[item_to_craft] += self.recipes[item_to_craft]['output'][item_to_craft]

            # if the agent craft stick before plank < 8
            if item_to_craft == 'stick' and self.inventory_items_quantity['plank'] < 8:
                # print('Don\'t craft stick before 8 plank ...')
                pass
            else:
                reward = 10

        return reward

    def has_ingredients_to_craft(self, item_to_craft):

        # Check if there are enough ingredients in the inventory
        have_all_ingredients = {}
        for item in self.recipes[item_to_craft]['input']:
            if self.inventory_items_quantity[item] >= self.recipes[item_to_craft]['input'][item]:
                have_all_ingredients[item] = True
            else:
                have_all_ingredients[item] = False

        # If there is not enough ingredients in the inventory
        if False in have_all_ingredients.values():
            return False
        else:
            return True

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
        plt.imshow(self.map, cMAP=color_map, vmin=0, vmax=len(self.items_id))
        plt.arrow(c, r, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(self.map_size, self.map_size // 2, 'EAST', rotation=90)
        # plt.colorbar()
        # plt.grid()

        info = '\n'.join(["               Info:             ",
                          "Env: " + self.env_name,
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
