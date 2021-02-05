import copy

import numpy as np

import gym
from gym import error, spaces, utils


class AxeEasy(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the inventory
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_name = axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id) + 1)
        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

    def reset(self):

        obs = self.env.reset()

        self.env.inventory_items_quantity.update({self.axe_name: 1})

        return obs

    def step(self, action_id):

        old_inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)

        observation, reward, done, info = self.env.step(action_id)

        if action_id == self.actions_id['Break']:
            if old_inventory_items_quantity != self.env.inventory_items_quantity:
                if self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'wooden_axe':
                    info['step_cost'] = info['step_cost'] * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    info['step_cost'] = info['step_cost'] * 0.25  # 900.0
                self.env.last_step_cost = info['step_cost']

        return observation, reward, done, info


class AxeMedium(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the map
    When the agent goes near axe, axe gets into the inventory
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_name = axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.add_new_items({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

    def step(self, action_id):

        old_inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)

        observation, reward, done, info = self.env.step(action_id)

        if action_id == self.actions_id['Break']:
            if old_inventory_items_quantity != self.env.inventory_items_quantity:
                if self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'wooden_axe':
                    info['step_cost'] = info['step_cost'] * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    info['step_cost'] = info['step_cost'] * 0.25  # 900.0
                self.env.last_step_cost = info['step_cost']

        return observation, reward, done, info


class AxeHard(gym.core.Wrapper):
    """
    Novelty wrapper to add a new recipe and action to craft axe
    When the agent crafts axe, it goes in the inventory
    Using axe reduces the step_cost to when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_material = axe_material
        self.axe_name = self.axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id) + 1)
        self.env.inventory_items_quantity.update({self.axe_name: 0})
        self.env.entities.add(self.axe_name)

        # Action Space
        if self.axe_material == 'wooden':
            self.env.recipes.update({self.axe_name: {'input': {'stick': 2, 'plank': 3}, 'output': {self.axe_name: 1}}})
        elif self.axe_material == 'iron':
            self.env.add_new_items({'iron': 3})
            self.env.recipes.update({self.axe_name: {'input': {'stick': 2, 'iron': 3}, 'output': {self.axe_name: 1}}})
        # self.action_craft_str.update({len(self.action_str): 'Craft_' + self.axe_name})
        self.env.actions_id.update({'Craft_' + self.axe_name: len(self.env.actions_id)})
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.action_space = spaces.Discrete(len(self.env.actions_id))

    def step(self, action_id):

        old_inventory_items_quantity = copy.deepcopy(self.env.inventory_items_quantity)

        observation, reward, done, info = self.env.step(action_id)

        # Craft___axe
        if action_id == len(self.env.actions_id) - 2:
            reward, result, step_cost, message = self.craft(self.axe_name)
            observation = self.env.get_observation()
            info = {'result': result, 'step_cost': step_cost, 'message': message}
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
        # Break
        elif action_id == self.actions_id['Break']:
            if old_inventory_items_quantity != self.env.inventory_items_quantity:
                if self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'wooden_axe':
                    info['step_cost'] = info['step_cost'] * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    info['step_cost'] = info['step_cost'] * 0.25  # 900.0
                self.env.last_step_cost = info['step_cost']

        return observation, reward, done, info

    def craft(self, item_to_craft):

        reward = -1  # default reward to craft in a wrong way
        result = True
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
            result = False
            message = "Missing items: "
            if item_to_craft == 'tree_tap':
                step_cost = 360.0
            elif item_to_craft == 'pogo_stick':
                step_cost = 480.0
            for item in have_all_ingredients:
                if not have_all_ingredients[item]:
                    message += str(self.env.recipes[item_to_craft]['input'][item]) + ' ' + item + ', '
            return reward, result, step_cost, message[:-2]
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
                    elif item_to_craft == self.axe_name:
                        step_cost = 600.0
                    result = False
                    message = 'Need to be in front of crafting_table'
                    return reward, result, step_cost, message

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
            elif item_to_craft == self.axe_name:
                step_cost = 6000.0

            message = 'Crafted ' + item_to_craft

            return reward, result, step_cost, message


class AxetoBreakEasy(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the inventory and requiring axe to break items
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_name = axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id) + 1)
        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

    def reset(self):

        obs = self.env.reset()

        self.env.inventory_items_quantity.update({self.axe_name: 1})

        return obs

    def step(self, action_id):

        if action_id == self.actions_id['Break']:
            self.env.last_action = list(self.actions_id.keys())[list(self.actions_id.values()).index(action_id)]

            reward = -1  # default reward
            result = True
            step_cost = 3600.0
            message = ''

            self.env.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if self.env.block_in_front_str not in self.env.unbreakable_items:
                if self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'wooden_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    if self.env.block_in_front_str == 'tree_log':
                        reward = 10
                    else:
                        reward = -10  # break something else

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    if self.env.block_in_front_str == 'tree_log':
                        reward = 10
                    else:
                        reward = -10  # break something else

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    result = False
                    message = "Cannot break without " + self.axe_name + " selected"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            observation = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = 50
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done
        else:
            observation, reward, done, info = self.env.step(action_id)

        return observation, reward, done, info


class AxetoBreakMedium(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the map
    When the agent goes near axe, axe gets into the inventory
    Axe is required to break items
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_name = axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.add_new_items({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

    def step(self, action_id):

        if action_id == self.actions_id['Break']:
            self.env.last_action = list(self.actions_id.keys())[list(self.actions_id.values()).index(action_id)]

            reward = -1  # default reward
            result = True
            step_cost = 3600.0
            message = ''

            self.env.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if self.env.block_in_front_str not in self.env.unbreakable_items:
                if self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'wooden_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    if self.env.block_in_front_str == 'tree_log':
                        reward = 10
                    else:
                        reward = -10  # break something else

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    if self.env.block_in_front_str == 'tree_log':
                        reward = 10
                    else:
                        reward = -10  # break something else

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    result = False
                    message = "Cannot break without " + self.axe_name + " selected"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            observation = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = 50
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done
        else:
            observation, reward, done, info = self.env.step(action_id)

        return observation, reward, done, info


class AxetoBreakHard(gym.core.Wrapper):
    """
    Novelty wrapper to add a new recipe and action to craft axe
    Agent starts with ingredients to craft an axe
    When the agent crafts axe, it goes in the inventory
    Axe is required to break items
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_material = axe_material
        self.axe_name = self.axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id) + 1)
        if self.axe_material == 'wooden':
            self.env.inventory_items_quantity.update({'wooden_axe': 0, 'stick': 2, 'plank': 3})
            self.env.recipes.update({self.axe_name: {'input': {'stick': 2, 'plank': 3}, 'output': {self.axe_name: 1}}})
        elif self.axe_material == 'iron':
            self.env.items.add('iron')
            self.env.items_id.setdefault('iron', len(self.items_id) + 1)
            self.env.inventory_items_quantity.update({'iron_axe': 0, 'stick': 2, 'iron': 3})
            self.env.recipes.update({self.axe_name: {'input': {'stick': 2, 'iron': 3}, 'output': {self.axe_name: 1}}})
        self.env.entities.add(self.axe_name)

        # Action Space
        # self.action_craft_str.update({'Craft_' + self.axe_name: len(self.action_str)})
        self.env.actions_id.update({'Craft_' + self.axe_name: len(self.env.actions_id)})
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.action_space = spaces.Discrete(len(self.env.actions_id))

    def reset(self):

        obs = self.env.reset()

        if self.axe_material == 'wooden':
            self.env.inventory_items_quantity.update({'wooden_axe': 0, 'stick': 2, 'plank': 3})
        elif self.axe_material == 'iron':
            self.env.inventory_items_quantity.update({'iron_axe': 0, 'stick': 2, 'iron': 3})

        return obs

    def step(self, action_id):

        # Craft___axe
        if action_id == len(self.env.actions_id) - 2:
            reward, result, step_cost, message = self.craft(self.axe_name)
            observation = self.env.get_observation()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = 50
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done
        elif action_id == self.actions_id['Break']:
            self.env.last_action = list(self.actions_id.keys())[list(self.actions_id.values()).index(action_id)]

            reward = -1  # default reward
            result = True
            step_cost = 3600.0
            message = ''

            self.env.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if self.env.block_in_front_str not in self.env.unbreakable_items:
                if self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'wooden_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    if self.env.block_in_front_str == 'tree_log':
                        reward = 10
                    else:
                        reward = -10  # break something else

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    if self.env.block_in_front_str == 'tree_log':
                        reward = 10
                    else:
                        reward = -10  # break something else

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    result = False
                    message = "Cannot break without " + self.axe_name + " selected"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            observation = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = 50
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done
        else:
            observation, reward, done, info = self.env.step(action_id)

        return observation, reward, done, info

    def craft(self, item_to_craft):

        reward = -1  # default reward to craft in a wrong way
        result = True
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
            result = False
            message = "Missing items: "
            if item_to_craft == 'tree_tap':
                step_cost = 360.0
            elif item_to_craft == 'pogo_stick':
                step_cost = 480.0
            for item in have_all_ingredients:
                if not have_all_ingredients[item]:
                    message += str(self.env.recipes[item_to_craft]['input'][item]) + ' ' + item + ', '
            return reward, result, step_cost, message[:-2]
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
                    elif item_to_craft == self.axe_name:
                        step_cost = 600.0
                    result = False
                    message = 'Need to be in front of crafting_table'
                    return reward, result, step_cost, message

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
            elif item_to_craft == self.axe_name:
                step_cost = 6000.0

            message = 'Crafted ' + item_to_craft

            return reward, result, step_cost, message


class Fence(gym.core.Wrapper):
    """
    Novelty wrapper to add fence around items in the map
    """

    def __init__(self, env, difficulty, fence_material):
        super().__init__(env)

        self.fence_name = fence_material + '_fence'  # oak_fence, jungle_fence
        self.env.items.add(self.fence_name)
        self.env.items_id.setdefault(self.fence_name, len(self.items_id) + 1)
        self.env.select_actions_id.update({'Select_' + self.fence_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

        if difficulty == 'easy':
            self.fence_percent_range = (20, 50)
        elif difficulty == 'medium':
            self.fence_percent_range = (50, 90)
        else:
            self.fence_percent_range = (90, 100)

    def reset(self):

        self.env.reset()

        result = np.array(np.where((self.env.map != 0) & (self.env.map != self.env.items_id['wall'])))

        # Shuffling locations in result
        indices = np.arange(len(result[0]))
        np.random.shuffle(indices)
        result[0] = result[0][indices]
        result[1] = result[1][indices]

        fence_percent = np.random.randint(low=self.fence_percent_range[0], high=self.fence_percent_range[1], size=1)[0]
        for i in range(int(np.ceil(len(result[0]) * (fence_percent / 100)))):
            r, c = result[0][i], result[1][i]
            self.env.add_fence_around((r, c), self.fence_name)

        # Update after each reset
        observation = self.get_observation()
        self.update_block_in_front()

        return observation


class AddItem(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item in the map
    """

    def __init__(self, env, difficulty, item_to_add='paper'):
        super().__init__(env)

        self.item_to_add = item_to_add
        self.env.items.add(self.item_to_add)
        self.env.items_id.setdefault(self.item_to_add, len(self.items_id) + 1)
        # self.env.entities.add(self.item_to_add)
        self.env.select_actions_id.update({'Select_' + self.item_to_add: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

        if difficulty == 'easy':
            self.item_percent_range = (1, 10)
        elif difficulty == 'medium':
            self.item_percent_range = (10, 20)
        else:
            self.item_percent_range = (20, 30)

    def reset(self):

        self.env.reset()

        result = np.array(np.where(self.env.map == 0))

        # Shuffling locations in result
        indices = np.arange(len(result[0]))
        np.random.shuffle(indices)
        result[0] = result[0][indices]
        result[1] = result[1][indices]
        item_percent = np.random.randint(low=self.item_percent_range[0], high=self.item_percent_range[1], size=1)[0]
        for i in range(int(np.ceil(len(result[0]) * (item_percent / 100)))):
            r, c = result[0][i], result[1][i]
            if (r, c) != self.env.agent_location:
                self.env.map[r][c] = self.items_id[self.item_to_add]

        # Update after each reset
        observation = self.get_observation()
        self.update_block_in_front()

        return observation


class ReplaceItem(gym.core.Wrapper):
    """
    Novelty wrapper to replace an item with another
    """

    def __init__(self, env, difficulty, item_to_replace='wall', item_to_replace_with='brick'):
        super().__init__(env)

        self.item_to_replace = item_to_replace
        self.item_to_replace_with = item_to_replace_with
        assert self.item_to_replace in self.env.items_id, "Item to replace (" + self.item_to_replace + \
                                                          ") is not in the original map"
        assert self.item_to_replace_with not in self.env.items_id, "Item to replace with (" + self.item_to_replace_with\
                                                                   + ") should be a new item"

        self.env.items.add(self.item_to_replace_with)
        self.env.items_id.setdefault(self.item_to_replace_with, len(self.items_id) + 1)
        # self.env.entities.add(self.item_to_replace_with)
        self.env.select_actions_id.update({'Select_' + self.item_to_replace_with: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

        if self.item_to_replace == 'wall':
            self.env.unbreakable_items.add(self.item_to_replace_with)

        if difficulty == 'easy':
            self.item_percent_range = (5, 20)
        elif difficulty == 'medium':
            self.item_percent_range = (40, 90)
        else:
            self.item_percent_range = (99, 100)

    def reset(self):

        self.env.reset()

        result = np.array(np.where(self.env.map == self.env.items_id[self.item_to_replace]))

        # Shuffling locations in result
        indices = np.arange(len(result[0]))
        np.random.shuffle(indices)
        result[0] = result[0][indices]
        result[1] = result[1][indices]

        item_percent = np.random.randint(low=self.item_percent_range[0], high=self.item_percent_range[1], size=1)[0]
        for i in range(int(np.ceil(len(result[0]) * (item_percent / 100)))):
            r, c = result[0][i], result[1][i]
            if (r, c) != self.env.agent_location:
                self.env.map[r][c] = self.items_id[self.item_to_replace_with]

        # Update after each reset
        observation = self.get_observation()
        self.update_block_in_front()

        return observation


def remap_action_difficulty(env, difficulty='hard'):
    """
    Remap actions randomly

    """

    if difficulty == 'easy':
        env.manipulation_actions_id = env.remap_action(env.manipulation_actions_id, 0)
        env.actions_id.update(env.manipulation_actions_id)
    elif difficulty == 'medium':
        env.manipulation_actions_id = env.remap_action(env.manipulation_actions_id, 0)
        env.craft_actions_id = env.remap_action(env.craft_actions_id, len(env.manipulation_actions_id))
        env.actions_id.update(env.manipulation_actions_id)
        env.actions_id.update(env.craft_actions_id)
    else:
        env.actions_id = env.remap_action(env.actions_id, 0)
        env.craft_actions_id = {action: env.actions_id[action] for action in env.actions_id if
                                action.startswith('Craft')}
        env.select_actions_id = {action: env.actions_id[action] for action in env.actions_id if
                                 action.startswith('Select')}

    return env


# Novelty without difficulty types:

class BlockItem(gym.core.Wrapper):
    """
    Novelty wrapper to block crafting_table from tree_log when rubber is extracted
    """

    def __init__(self, env):
        super().__init__(env)

        self.items_to_block = 'crafting_table'
        self.item_to_block_from = 'tree_log'

        self.env.items.add('fence')
        self.env.items_id.setdefault('fence', len(self.items_id) + 1)

    def step(self, action_id):

        old_rubber_quantity = copy.deepcopy(self.env.inventory_items_quantity['rubber'])

        observation, reward, done, info = self.env.step(action_id)

        # Extract_rubber
        if action_id == self.actions_id['Extract_rubber']:
            if old_rubber_quantity < self.env.inventory_items_quantity['rubber']:
                # Block by self.item_to_block_from
                # self.env.block_items(item_to_block=self.items_to_block, item_to_block_from=self.item_to_block_from)

                # Block by fence
                result = np.where(self.env.map == self.env.items_id[self.items_to_block])
                for i in range(len(result[0])):
                    r, c = result[0][i], result[1][i]
                    self.env.add_fence_around((r, c))

        return observation, reward, done, info


class AddChopAction(gym.core.Wrapper):
    """
    Novelty wrapper to add chop action
    """

    def __init__(self, env):
        super().__init__(env)

        self.env.manipulation_actions_id['chop'] = len(self.actions_id)
        self.env.actions_id.update(self.manipulation_actions_id)
        self.action_space = spaces.Discrete(len(self.actions_id))

    def step(self, action_id):

        if action_id == self.actions_id['chop']:

            self.env.last_action = list(self.actions_id.keys())[list(self.actions_id.values()).index(action_id)]

            reward = -1  # default reward
            result = True
            step_cost = 3600.0 * 1.2
            message = ''

            self.env.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if self.block_in_front_str not in self.unbreakable_items:
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0
                self.inventory_items_quantity[self.block_in_front_str] += 1 * 2

                if self.block_in_front_str == 'tree_log':
                    reward = 10
                else:
                    reward = -10  # break something else
            else:
                result = False
                message = "Cannot chop " + self.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            observation = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = 50
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done
        else:
            observation, reward, done, info = self.env.step(action_id)

        return observation, reward, done, info

class BreakIncrease(gym.core.Wrapper):
    """
    Novelty wrapper to get 2 items in inventory when the agent break that item instead of 1
    itemtobreakmore: apply this to only itemtobreakmore or if itemtobreakmore == '', then to all items
    """

    def __init__(self, env, itemtobreakmore=''):
        super().__init__(env)

        self.itemtobreakmore = itemtobreakmore

    def step(self, action_id):

        if action_id == self.actions_id['Break']:
            self.env.last_action = list(self.actions_id.keys())[list(self.actions_id.values()).index(action_id)]

            reward = -1  # default reward
            result = True
            step_cost = 3600.0
            message = ''

            self.env.update_block_in_front()
            # If block in front is not air and wall, place the block in front in inventory
            if self.env.block_in_front_str not in self.env.unbreakable_items:
                block_r, block_c = self.env.block_in_front_location
                self.env.map[block_r][block_c] = 0
                if self.itemtobreakmore == self.env.block_in_front_str:
                    self.env.inventory_items_quantity[self.itemtobreakmore] += 2
                elif self.itemtobreakmore == '':
                    self.env.inventory_items_quantity[self.block_in_front_str] += 2
                else:
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                if self.env.block_in_front_str == 'tree_log':
                    reward = 10
                else:
                    reward = -10  # break something else
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            observation = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = 50
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done
        else:
            observation, reward, done, info = self.env.step(action_id)

        return observation, reward, done, info

#################### Novelty Helper ####################

def inject_novelty(env, difficulty, novelty_name, novelty_arg1, novelty_arg2):

    if novelty_name == 'addchop':
        return AddChopAction(env)
    elif novelty_name == 'breakincrease':
        if novelty_arg1:
            env = BreakIncrease(env, novelty_arg1)
        else:
            env = BreakIncrease(env)

    if difficulty == 'easy':
        if novelty_name == 'axe':
            env = AxeEasy(env, novelty_arg1)
        elif novelty_name == 'axetobreak':
            env = AxetoBreakEasy(env, novelty_arg1)
        elif novelty_name == 'fence':
            env = Fence(env, difficulty, novelty_arg1)
        elif novelty_name == 'additem':
            env = AddItem(env, difficulty, novelty_arg1)
        elif novelty_name == 'replaceitem':
            env = ReplaceItem(env, difficulty, novelty_arg1, novelty_arg2)
        elif novelty_name == 'remapaction':
            env = remap_action_difficulty(env, difficulty)
    elif difficulty == 'medium':
        if novelty_name == 'axe':
            env = AxeMedium(env, novelty_arg1)
        elif novelty_name == 'axetobreak':
            env = AxetoBreakMedium(env, novelty_arg1)
        elif novelty_name == 'fence':
            env = Fence(env, difficulty, novelty_arg1)
        elif novelty_name == 'additem':
            env = AddItem(env, difficulty, novelty_arg1)
        elif novelty_name == 'replaceitem':
            env = ReplaceItem(env, difficulty, novelty_arg1, novelty_arg2)
        elif novelty_name == 'remapaction':
            env = remap_action_difficulty(env, difficulty)
    elif difficulty == 'hard':
        if novelty_name == 'axe':
            env = AxeHard(env, novelty_arg1)
        elif novelty_name == 'axetobreak':
            env = AxetoBreakHard(env, novelty_arg1)
        elif novelty_name == 'fence':
            env = Fence(env, difficulty, novelty_arg1)
        elif novelty_name == 'additem':
            env = AddItem(env, difficulty, novelty_arg1)
        elif novelty_name == 'replaceitem':
            env = ReplaceItem(env, difficulty, novelty_arg1, novelty_arg2)
        elif novelty_name == 'remapaction':
            env = remap_action_difficulty(env, difficulty)

    return env
