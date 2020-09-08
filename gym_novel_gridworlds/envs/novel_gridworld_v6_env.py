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


class NovelGridworldV6Env(gym.Env):
    # metadata = {'render.modes': ['human']}
    """
    Goal: Craft 1 pogo_stick
    State: map, agent_location, agent_facing_id, inventory_items_quantity
    Action: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break', 4: 'Place_tree_tap', 5: 'Extract_rubber',
            6: 'Craft_plank', 7: 'Craft_stick', 8: 'Craft_tree_tap', 9: 'Craft_pogo_stick'}
    """

    def __init__(self, env=None):
        # NovelGridworldV6Env attributes
        self.env_name = 'NovelGridworld-v6'
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
        self.items_quantity = {'crafting_table': 1, 'tree_log': 5}
        self.inventory_items_quantity = {item: 0 for item in self.items}
        self.selected_item = ''
        self.entities = ['']
        self.available_locations = []  # locations that do not have item placed
        self.not_available_locations = []  # locations that have item placed or are above, below, left, right to an item

        # Action Space
        self.action_str = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break', 4: 'Place_tree_tap', 5: 'Extract_rubber',
                           6: 'Craft_plank', 7: 'Craft_stick', 8: 'Craft_tree_tap', 9: 'Craft_pogo_stick'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.recipes = {'pogo_stick': {'input': {'stick': 4, 'plank': 2, 'rubber': 1}, 'output': {'pogo_stick': 1}},
                        'stick': {'input': {'plank': 2}, 'output': {'stick': 4}},
                        'plank': {'input': {'tree_log': 1}, 'output': {'plank': 4}},
                        'tree_tap': {'input': {'plank': 5, 'stick': 1}, 'output': {'tree_tap': 1}},
                        'crafting_table': {'input': {'plank': 4}, 'output': {'crafting_table': 1}}}
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken
        self.last_step_cost = 0  # last received step_cost

        # Observation Space
        self.max_items = 20
        self.observation_space = spaces.Box(low=0, high=self.max_items, shape=(self.map_size, self.map_size, 1))
        self.observation_space = spaces.Dict({'map': self.observation_space})

        # Reward
        self.last_reward = 0  # last received reward

        self.last_done = False  # last done

    def reset(self, map_size=None, items_id=None, items_quantity=None):

        print("RESETTING " + self.env_name + " ...")
        if self.env is not None:
            print("RESTORING " + self.env_name + " ...")
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
        self.selected_item = ''
        self.available_locations = []
        self.not_available_locations = []
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken
        self.last_step_cost = 0  # last received step_cost
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
        observation: map, agent_location, agent_facing_id, inventory_items_quantity
        :return: observation
        """

        assert not self.max_items < len(self.items), "Cannot have more than " + str(self.max_items) + " items"

        observation = {'map': self.map,
                       'agent_location': self.agent_location,
                       'agent_facing_id': self.agent_facing_id,
                       'inventory_items_quantity': self.inventory_items_quantity,
                       }

        return observation

    def step(self, action):
        """
        Actions: {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Break', 4: 'Place_tree_tap', 5: 'Extract_rubber',
                6: 'Craft_plank', 7: 'Craft_stick', 8: 'Craft_tree_tap', 9: 'Craft_pogo_stick'}
        """

        self.last_action = action
        r, c = self.agent_location

        reward = -1  # default reward
        step_cost = 0  # default step_cost
        message = ''

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

            step_cost = 27.906975
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

            step_cost = 24.0
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

            step_cost = 24.0
        # Break
        elif action == 3:
            self.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if self.block_in_front_str not in ['air', 'wall']:
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0

                if self.block_in_front_str == 'tree_log':
                    reward = 10
                else:
                    reward = -10  # break something else
                self.inventory_items_quantity[self.block_in_front_str] += 1
            else:
                message = "Cannot break " + self.block_in_front_str

            step_cost = 3600.0
        # Place_tree_tap
        elif action == 4:
            reward = -1  # default reward to Place_tree_tap

            if self.inventory_items_quantity['tree_tap'] >= 1 and self.block_in_front_str == 'air':
                r, c = self.block_in_front_location
                self.map[r][c] = self.items_id['tree_tap']  # Place_tree_tap
                self.inventory_items_quantity['tree_tap'] -= 1

                # Make sure that block_in_front_location is next to a tree
                block_in_front_next_to_tree = self.is_block_in_front_next_to('tree_log')
                if block_in_front_next_to_tree:
                    reward = 20

            step_cost = 300.0
        # Extract_rubber
        elif action == 5:
            reward = -1  # default reward
            step_cost = 120.0  # default step_cost

            # Make sure that block_in_front_location is next to a tree
            block_in_front_next_to_tree = self.is_block_in_front_next_to('tree_log')

            if block_in_front_next_to_tree and self.block_in_front_str == 'tree_tap':
                self.inventory_items_quantity['rubber'] += 1  # Extract_rubber
                reward = 15
                step_cost = 50000
        # Craft_plank
        elif action == 6:
            item_to_craft = 'plank'
            reward, step_cost, message = self.craft(item_to_craft)
        # Craft_stick
        elif action == 7:
            item_to_craft = 'stick'
            reward, step_cost, message = self.craft(item_to_craft)
        # Craft_tree_tap
        elif action == 8:
            item_to_craft = 'tree_tap'
            reward, step_cost, message = self.craft(item_to_craft)
        # Craft_pogo_stick
        elif action == 9:
            item_to_craft = 'pogo_stick'
            reward, step_cost, message = self.craft(item_to_craft)

        # Update after each step
        self.grab_entities()
        observation = self.get_observation()
        self.update_block_in_front()

        done = False
        if self.inventory_items_quantity['pogo_stick'] >= 1:
            reward = 50
            done = True

        info = {'step_cost': step_cost, 'message': message}

        # Update after each step
        self.step_count += 1
        self.last_step_cost = step_cost
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

    def is_block_in_front_next_to(self, item):

        self.update_block_in_front()
        r, c = self.block_in_front_location

        # Make sure that block_in_front_location is next to item
        block_in_front_next_to_item = False
        # NORTH
        if (0 <= (r - 1) <= self.map_size - 1) and self.map[r - 1][c] == self.items_id[item]:
            block_in_front_next_to_item = True
        # SOUTH
        elif (0 <= (r + 1) <= self.map_size - 1) and self.map[r + 1][c] == self.items_id[item]:
            block_in_front_next_to_item = True
        # WEST
        elif (0 <= (c - 1) <= self.map_size - 1) and self.map[r][c - 1] == self.items_id[item]:
            block_in_front_next_to_item = True
        # EAST
        elif (0 <= (c + 1) <= self.map_size - 1) and self.map[r][c + 1] == self.items_id[item]:
            block_in_front_next_to_item = True

        return block_in_front_next_to_item

    def craft(self, item_to_craft):

        reward = -1  # default reward to craft in a wrong way
        step_cost = 0  # default step_cost
        message = ''

        # Check if there are enough ingredients in the inventory
        have_all_ingredients = {}
        for item in self.recipes[item_to_craft]['input']:
            if item in self.inventory_items_quantity and self.inventory_items_quantity[item] >= \
                    self.recipes[item_to_craft]['input'][item]:
                have_all_ingredients[item] = True
            else:
                have_all_ingredients[item] = False

        # If there is not enough ingredients in the inventory
        if False in have_all_ingredients.values():
            message = "Missing items: "
            if item_to_craft == 'tree_tap':
                step_cost = 360.0
            elif item_to_craft == 'pogo_stick':
                step_cost = 480.0
            for item in have_all_ingredients:
                if not have_all_ingredients[item]:
                    message += str(self.recipes[item_to_craft]['input'][item]) + ' ' + item + ', '
            return reward, step_cost, message[:-2]
        # Craft
        else:
            # If more than 1 ingredient needed, agent needs to be in front of crafting_table
            if len(self.recipes[item_to_craft]['input']) > 1:
                self.update_block_in_front()
                if not self.block_in_front_str == 'crafting_table':
                    if item_to_craft == 'tree_tap':
                        step_cost = 720.0
                    elif item_to_craft == 'pogo_stick':
                        step_cost = 840.0
                    message = 'Need to be in front of crafting_table'
                    return reward, step_cost, message

            reward = 10  # default reward to craft in a good way

            # Reduce ingredients from the inventory
            for item in self.recipes[item_to_craft]['input']:
                self.inventory_items_quantity[item] -= self.recipes[item_to_craft]['input'][item]
            # Add item_to_craft in the inventory
            self.inventory_items_quantity[item_to_craft] += self.recipes[item_to_craft]['output'][item_to_craft]

            if item_to_craft == 'plank':
                step_cost = 1200.0
            elif item_to_craft == 'stick':
                step_cost = 2400.0
            elif item_to_craft == 'tree_tap':
                step_cost = 7200.0
            elif item_to_craft == 'pogo_stick':
                step_cost = 8400.0

            message = 'Crafted ' + item_to_craft

            return reward, step_cost, message

    def add_new_items(self, new_items_quantity):

        for item in new_items_quantity:
            self.items.append(item)
            self.items_id.setdefault(item, len(self.items_id) + 1)
            self.items_quantity.update({item: new_items_quantity[item]})
        self.reset()

    def block_items(self, item_to_block, item_to_block_from):
        """
        Add item_to_block_from around item_to_block
        """

        result = np.where(self.map == self.items_id[item_to_block])
        for i in range(len(result[0])):
            r, c = result[0][i], result[1][i]
            # NORTH
            if (0 <= (r - 1) <= self.map_size - 1) and self.map[r - 1][c] == 0 and (r - 1, c) != self.agent_location:
                self.map[r - 1][c] = self.items_id[item_to_block_from]
            # SOUTH
            if (0 <= (r + 1) <= self.map_size - 1) and self.map[r + 1][c] == 0 and (r + 1, c) != self.agent_location:
                self.map[r + 1][c] = self.items_id[item_to_block_from]
            # WEST
            if (0 <= (c - 1) <= self.map_size - 1) and self.map[r][c - 1] == 0 and (r, c - 1) != self.agent_location:
                self.map[r][c - 1] = self.items_id[item_to_block_from]
            # EAST
            if (0 <= (c + 1) <= self.map_size - 1) and self.map[r][c + 1] == 0 and (r, c + 1) != self.agent_location:
                self.map[r][c + 1] = self.items_id[item_to_block_from]

    def grab_entities(self):

        r, c = self.agent_location

        for r_ent in [r - 1, r, r + 1]:
            for c_ent in [c - 1, c, c + 1]:
                ent_id_rc = self.map[r_ent][c_ent]

                # If there's an entity next to agent
                if ent_id_rc != 0:
                    item = list(self.items_id.keys())[list(self.items_id.values()).index(ent_id_rc)]
                    if item in self.entities:
                        self.map[r_ent][c_ent] = 0
                        self.inventory_items_quantity[item] += 1

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
                          "Steps: " + str(self.step_count),
                          "Agent Facing: " + self.agent_facing_str,
                          "Action: " + self.action_str[self.last_action],
                          "Reward: " + str(self.last_reward),
                          "Step Cost: " + str(self.last_step_cost),
                          "Done: " + str(self.last_done)])
        props = dict(boxstyle='round', facecolor='w', alpha=0.2)
        plt.text(-(self.map_size // 2) - 0.5, 2.25, info, fontsize=10, bbox=props)  # x, y

        if self.last_done:
            you_win = "YOU WIN " + self.env_name + "!!!"
            props = dict(boxstyle='round', facecolor='w', alpha=1)
            plt.text(0 - 0.1, (self.map_size // 2), you_win, fontsize=18, bbox=props)
            if self.inventory_items_quantity['pogo_stick'] >= 1:
                you_win = "YOU CRAFTED POGO_STICK!!!"
                props = dict(boxstyle='round', facecolor='w', alpha=1)
                plt.text(0 - 0.1, (self.map_size // 2) + 1, you_win, fontsize=18, bbox=props)

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
