import copy

import gym
from gym import error, spaces, utils


class Level1Easy(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the inventory and select it
    Using axe reduces the step_cost to 50% (1800.0) when Break action is used
    """

    def __init__(self, env):
        super().__init__(env)

        self.env.items.append('axe')
        self.env.items_id.setdefault('axe', len(self.items_id) + 1)
        self.env.inventory_items_quantity.update({'axe': 1})
        self.env.selected_item = 'axe'

    def reset(self):

        obs = self.env.reset()

        self.env.inventory_items_quantity.update({'axe': 1})
        self.env.selected_item = 'axe'

        return obs

    def step(self, action):

        old_inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)

        observation, reward, done, info = self.env.step(action)

        # Break
        if action == 3:
            if old_inventory_items_quantity != self.env.inventory_items_quantity and self.env.selected_item == 'axe':
                info['step_cost'] = info['step_cost'] * 0.5  # 1800.0
                self.env.last_step_cost = info['step_cost']

        return observation, reward, done, info


class Level1Medium(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the map
    When the agent goes near axe, axe gets into the inventory and gets selected
    Using axe reduces the step_cost to 50% (1800.0) when Break action is used
    """

    def __init__(self, env):
        super().__init__(env)

        self.env.add_new_items({'axe': 1})
        self.env.entities.append('axe')

    def step(self, action):

        old_inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)

        observation, reward, done, info = self.env.step(action)

        if self.env.inventory_items_quantity['axe'] >= 1:
            self.env.selected_item = 'axe'

        # Break
        if action == 3:
            if old_inventory_items_quantity != self.env.inventory_items_quantity and \
                    self.env.inventory_items_quantity['axe'] >= 1 and self.env.selected_item == 'axe':
                info['step_cost'] = info['step_cost'] * 0.5  # 1800.0
                self.env.last_step_cost = info['step_cost']

        return observation, reward, done, info


class Level1Hard(gym.core.Wrapper):
    """
    Novelty wrapper to add a new recipe and action to craft axe
    When the agent crafts axe, it goes in the inventory and gets selected
    Using axe reduces the step_cost to 50% (1800.0) when Break action is used
    """

    def __init__(self, env):
        super().__init__(env)

        # Action Space
        self.env.action_str.update({len(self.action_str): 'Craft_axe'})
        self.env.action_space = spaces.Discrete(len(self.action_str))
        self.env.recipes.update({'axe': {'input': {'stick': 2, 'plank': 3}, 'output': {'axe': 1}}})

    def step(self, action):

        old_inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)

        observation, reward, done, info = self.env.step(action)

        # Craft_axe
        if action == len(self.env.action_str) - 1:
            item_to_craft = 'axe'

            self.env.items.append(item_to_craft)
            self.env.items_id.setdefault(item_to_craft, len(self.items_id) + 1)
            self.env.inventory_items_quantity.update({item_to_craft: 0})
            self.env.entities.append(item_to_craft)

            reward, step_cost, message = self.craft(item_to_craft)
            observation = self.env.get_observation()
            info = {'step_cost': step_cost, 'message': message}
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward

            if self.env.inventory_items_quantity[item_to_craft] >= 1:
                self.env.selected_item = item_to_craft
        # Break
        elif action == 3:
            if old_inventory_items_quantity != self.env.inventory_items_quantity \
                    and 'axe' in self.env.inventory_items_quantity and self.env.inventory_items_quantity['axe'] >= 1\
                    and self.env.selected_item == 'axe':
                info['step_cost'] = info['step_cost'] * 0.5  # 1800.0
                self.env.last_step_cost = info['step_cost']

        return observation, reward, done, info

    def craft(self, item_to_craft):

        reward = -1  # default reward to craft in a wrong way
        step_cost = 0  # default step_cost
        message = ''

        # Check if there are enough ingredients in the inventory
        have_all_ingredients = {}
        for item in self.env.recipes[item_to_craft]['input']:
            if item in self.env.inventory_items_quantity and self.env.inventory_items_quantity[item] >= \
                    self.env.recipes[item_to_craft]['input'][item]:
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
                    message += str(self.env.recipes[item_to_craft]['input'][item]) + ' ' + item + ', '
            return reward, step_cost, message[:-2]
        # Craft
        else:
            # If more than 1 ingredient needed, agent needs to be in front of crafting_table
            if len(self.env.recipes[item_to_craft]['input']) > 1:
                self.env.update_block_in_front()
                if not self.env.block_in_front_str == 'crafting_table':
                    if item_to_craft == 'tree_tap':
                        step_cost = 720.0
                    elif item_to_craft == 'pogo_stick':
                        step_cost = 840.0
                    elif item_to_craft == 'axe':
                        step_cost = 600.0
                    message = 'Need to be in front of crafting_table'
                    return reward, step_cost, message

            reward = 10  # default reward to craft in a good way

            # Reduce ingredients from the inventory
            for item in self.env.recipes[item_to_craft]['input']:
                self.env.inventory_items_quantity[item] -= self.env.recipes[item_to_craft]['input'][item]
            # Add item_to_craft in the inventory
            self.env.inventory_items_quantity[item_to_craft] += self.env.recipes[item_to_craft]['output'][item_to_craft]

            if item_to_craft == 'plank':
                step_cost = 1200.0
            elif item_to_craft == 'stick':
                step_cost = 2400.0
            elif item_to_craft == 'tree_tap':
                step_cost = 7200.0
            elif item_to_craft == 'pogo_stick':
                step_cost = 8400.0
            elif item_to_craft == 'axe':
                step_cost = 6000.0

            message = 'Crafted ' + item_to_craft

            return reward, step_cost, message


class BlockItem(gym.core.Wrapper):
    """
    Novelty wrapper to block crafting_table from tree_log when rubber is extracted
    """

    def __init__(self, env):
        super().__init__(env)

        self.items_to_block = 'crafting_table'
        self.item_to_block_from = 'tree_log'

    def step(self, action):

        old_rubber_quantity = copy.deepcopy(self.env.inventory_items_quantity['rubber'])

        observation, reward, done, info = self.env.step(action)

        # Extract_rubber
        if action == 5:
            if old_rubber_quantity < self.env.inventory_items_quantity['rubber']:
                self.env.block_items(item_to_block=self.items_to_block, item_to_block_from=self.item_to_block_from)

        return observation, reward, done, info
