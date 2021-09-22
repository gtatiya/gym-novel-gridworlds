import copy

import numpy as np

import gym
from gym import error, spaces, utils


class AxeEasy(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the inventory
    Using axe reduces the step_cost when Break action is used
    With optional arg breakincrease, the agent will get 2 items in inventory after break action instead of 1
    """

    def __init__(self, env, axe_material, breakincrease='false'):
        super().__init__(env)

        self.axe_name = axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

        self.breakincrease = breakincrease

    def reset(self):

        obs = self.env.reset()

        self.env.inventory_items_quantity.update({self.axe_name: 1})

        return obs

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use breakincrease novelty_arg2 because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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
                    if self.breakincrease == 'true':
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 2
                    else:
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    if self.breakincrease == 'true':
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 2
                    else:
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class AxeMedium(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the map
    When the agent goes near axe, axe gets into the inventory
    Using axe reduces the step_cost when Break action is used
    With optional arg breakincrease, the agent will get 2 items in inventory after break action instead of 1
    """

    def __init__(self, env, axe_material, breakincrease='false'):
        super().__init__(env)

        self.axe_name = axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.add_new_items({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

        self.breakincrease = breakincrease

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use breakincrease novelty_arg2 because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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
                    if self.breakincrease == 'true':
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 2
                    else:
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    if self.breakincrease == 'true':
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 2
                    else:
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class AxeHard(gym.core.Wrapper):
    """
    Novelty wrapper to add a new recipe and action to craft axe
    The ingredients of axe are placed in the map
    When the agent crafts axe, it goes in the inventory
    Using axe reduces the step_cost to when Break action is used
    With optional arg breakincrease, the agent will get 2 items in inventory after break action instead of 1
    """

    def __init__(self, env, axe_material, breakincrease='false'):
        super().__init__(env)

        self.axe_material = axe_material
        self.axe_name = self.axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.axe_name: 0})
        self.env.entities.add(self.axe_name)

        # Action Space
        if self.axe_material == 'wooden':
            axe_recipe = {'stick': 2, 'plank': 3}
        elif self.axe_material == 'iron':
            axe_recipe = {'stick': 2, 'iron': 3}
        # adding axe's ingredients to map
        for item in axe_recipe:
            if item in self.env.items:
                if item in self.items_quantity:
                    item_quantity = self.items_quantity[item]
                    self.env.items_quantity.update({item: item_quantity + axe_recipe[item]})
                else:
                    self.env.items_quantity.update({item: axe_recipe[item]})
            else:
                self.env.add_new_items({item: axe_recipe[item]})
        self.env.recipes.update({self.axe_name: {'input': axe_recipe, 'output': {self.axe_name: 1}}})

        self.env.craft_actions_id.update({'Craft_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update({'Craft_' + self.axe_name: len(self.env.actions_id)})
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.action_space = spaces.Discrete(len(self.env.actions_id))

        self.breakincrease = breakincrease

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Craft_' + self.axe_name in self.limited_actions_id,\
                "Cannot use AxeHard novelty because you do not have " + "Craft_" + self.axe_name + " in LimitActions"
            assert 'Break' in self.limited_actions_id, "Cannot use breakincrease novelty_arg2 because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Craft_' + self.axe_name]:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

            reward, result, step_cost, message = self.craft(self.axe_name)

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        elif action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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
                    if self.breakincrease == 'true':
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 2
                    else:
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    if self.breakincrease == 'true':
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 2
                    else:
                        self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info

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

        # If there are not enough ingredients in the inventory
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

            reward = self.reward_intermediate  # default reward to craft in a good way

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
        self.env.items_id.setdefault(self.axe_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

    def reset(self):

        obs = self.env.reset()

        self.env.inventory_items_quantity.update({self.axe_name: 1})

        return obs

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use axetobreak novelty because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    result = False
                    message = "Cannot break without " + self.axe_name + " selected"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


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

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use axetobreak novelty because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    result = False
                    message = "Cannot break without " + self.axe_name + " selected"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class AxetoBreakHard(gym.core.Wrapper):
    """
    Novelty wrapper to add a new recipe and action to craft axe
    Agent starts with ingredients to craft an axe in the inventory
    When the agent crafts axe, it goes in the inventory
    Axe is required to break items
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env, axe_material):
        super().__init__(env)

        self.axe_material = axe_material
        self.axe_name = self.axe_material + '_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.axe_name: 0})
        self.env.entities.add(self.axe_name)

        # Action Space
        if self.axe_material == 'wooden':
            axe_recipe = {'stick': 2, 'plank': 3}
        elif self.axe_material == 'iron':
            axe_recipe = {'stick': 2, 'iron': 3}
        for item in axe_recipe:
            if item not in self.env.items:
                self.env.items.add(item)
                self.env.items_id.setdefault(item, len(self.items_id))
        self.env.inventory_items_quantity.update(axe_recipe)
        self.env.recipes.update({self.axe_name: {'input': axe_recipe, 'output': {self.axe_name: 1}}})

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

        if hasattr(self, 'limited_actions_id'):
            assert 'Craft_' + self.axe_name in self.limited_actions_id,\
                "Cannot use AxetoBreakHard novelty because you do not have " + "Craft_" + self.axe_name + " in LimitActions"
            assert 'Break' in self.limited_actions_id, "Cannot use axetobreak novelty because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Craft_' + self.axe_name]:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

            reward, result, step_cost, message = self.craft(self.axe_name)

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        elif action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.25  # 900.0
                else:
                    result = False
                    message = "Cannot break without " + self.axe_name + " selected"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info

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

        # If there are not enough ingredients in the inventory
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

            reward = self.reward_intermediate  # default reward to craft in a good way

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


class CoinCraft(gym.core.Wrapper):
    """
    Novelty wrapper to require coins to craft; coins are generated on the map and
    must be picked up. Each craft action takes one coin, and cannot be performed with no coins
    """

    def __init__(self, env, difficulty):
        super().__init__(env)

        if self.env.map_size > 10 :
            self.num_coins = 10
        else :
            self.num_coins = 7

        self.coin_name = 'coin'

        self.env.add_new_items({self.coin_name: self.num_coins})
        self.env.entities.add(self.coin_name)
        self.env.select_actions_id.update({'Select_' + self.coin_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)

    def step(self, action_id):

        if action_id in self.env.craft_actions_id.values():
            craft_action = list(self.env.craft_actions_id.keys())[list(self.env.craft_actions_id.values()).index(action_id)]
            item_to_craft = '_'.join(craft_action.split('_')[1:])
            reward, result, step_cost, message = self.craft(item_to_craft)

            self.env.last_action = list(self.actions_id.keys())[list(self.actions_id.values()).index(action_id)]

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)

        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info

    def craft(self, item_to_craft):

        reward = -1  # default reward to craft in a wrong way
        result = True
        step_cost = 0  # default step_cost
        message = ''

        if self.env.inventory_items_quantity['coin'] == 0 :
            result = False
            message = "Not enough coins to craft"
        else :
            reward, result, step_cost, message = self.env.craft(item_to_craft)
            if result :
                self.env.inventory_items_quantity['coin'] -= 1

        return reward, result, step_cost, message

class Fence(gym.core.Wrapper):
    """
    Novelty wrapper to add fence around items in the map
    """

    def __init__(self, env, difficulty, fence_material):
        super().__init__(env)

        self.fence_name = fence_material + '_fence'  # oak_fence, jungle_fence
        self.env.items.add(self.fence_name)
        self.env.items_id.setdefault(self.fence_name, len(self.items_id))
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
        obs = self.get_observation()
        self.update_block_in_front()

        return obs


class FenceRestriction(gym.core.Wrapper):
    """
    Novelty wrapper to restrict breaking an item around fence until fence(s) are broken.
    All fences are always breakable.
    """

    def __init__(self, env, difficulty, fence_material):
        super().__init__(env)

        self.difficulty = difficulty
        self.env2 = Fence(env, 'medium', fence_material=fence_material)

    def reset(self):

        obs = self.env2.reset()

        return obs

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use fencerestriction novelty because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break'] and self.difficulty != 'easy':
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

            reward = -1  # default reward
            result = True
            step_cost = 3600.0
            message = ''

            self.env.update_block_in_front()
            if self.env.block_in_front_str not in self.env.unbreakable_items:
                if self.env.block_in_front_str == self.env2.fence_name:
                    # Fence is always breakable
                    obs, reward, done, info = self.env.step(action_id)
                else:
                    fence_restriction = False
                    if self.difficulty == 'medium':
                        # In medium, one side of the item to break must not have fence and agent must be on that side
                        r, c = self.agent_location
                        if self.agent_facing_str == 'NORTH' or self.agent_facing_str == 'SOUTH':
                            if self.map[r][c - 1] == self.items_id[self.env2.fence_name] or self.map[r][c + 1] == self.items_id[self.env2.fence_name]:
                                fence_restriction = True
                        elif self.agent_facing_str == 'WEST' or self.agent_facing_str == 'EAST':
                            if self.map[r - 1][c] == self.items_id[self.env2.fence_name] or self.map[r + 1][c] == self.items_id[self.env2.fence_name]:
                                fence_restriction = True
                    else:
                        # In hard, all the sides of the item to break must not have fence
                        r, c = self.block_in_front_location
                        for r_item in [r - 1, r, r + 1]:
                            for c_item in [c - 1, c, c + 1]:
                                if self.map[r_item][c_item] == self.items_id[self.env2.fence_name]:
                                    fence_restriction = True
                                    break

                    if not fence_restriction:
                        obs, reward, done, info = self.env.step(action_id)
                    else:
                        result = False
                        message = "Cannot break due to fence restriction"
            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class AddItem(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item in the map
    """

    def __init__(self, env, difficulty, item_to_add):
        super().__init__(env)

        self.item_to_add = item_to_add
        self.env.items.add(self.item_to_add)
        self.env.items_id.setdefault(self.item_to_add, len(self.items_id))
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
        obs = self.get_observation()
        self.update_block_in_front()

        return obs


class Crate(gym.core.Wrapper):
    """
    Novelty wrapper to add crate(s) in the map. When the crate is broken, some ingredients of the goal_item_to_craft
    gets in the inventory
    """

    def __init__(self, env, difficulty):
        super().__init__(env)

        self.env2 = AddItem(env, 'easy', item_to_add='crate')

        if difficulty == 'easy':
            item_percent_range = (99, 100)
        elif difficulty == 'medium':
            item_percent_range = (50, 90)
        else:
            item_percent_range = (10, 50)
        item_percent = np.random.randint(low=item_percent_range[0], high=item_percent_range[1], size=1)[0]

        total_ingredients = 0
        ingredients = []
        for item in self.recipes[self.goal_item_to_craft]['input']:
            total_ingredients += self.recipes[self.goal_item_to_craft]['input'][item]
            ingredients.append(item)

        crate_ingredients_num = int(np.ceil((item_percent / 100) * total_ingredients))

        self.crate_ingredients = []
        while crate_ingredients_num:
            item = np.random.choice(ingredients, size=1)[0]
            if self.crate_ingredients.count(item) < self.recipes[self.goal_item_to_craft]['input'][item]:
                self.crate_ingredients.append(item)
                crate_ingredients_num -= 1

    def reset(self):

        obs = self.env2.reset()

        return obs

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use crate novelty because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break'] and self.env.block_in_front_str == 'crate':
            for item in self.crate_ingredients:
                self.env.inventory_items_quantity[item] += 1
            obs, reward, done, info = self.env.step(action_id)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


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
        assert self.item_to_replace_with not in self.env.items_id, "Item to replace with (" + self.item_to_replace_with \
                                                                   + ") should be a new item"

        self.env.items.add(self.item_to_replace_with)
        self.env.items_id.setdefault(self.item_to_replace_with, len(self.items_id))
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
        obs = self.env.get_observation()
        self.update_block_in_front()

        return obs


class FireWall(gym.core.Wrapper):
    """
    Novelty wrapper to add fire_wall, agent dies when it's next to fire_wall
    """

    def __init__(self, env, difficulty='hard'):
        super().__init__(env)

        self.env2 = ReplaceItem(env, difficulty, item_to_replace='wall', item_to_replace_with='fire_wall')

    def reset(self):

        obs = self.env2.reset()

        return obs

    def step(self, action_id):

        obs, reward, done, info = self.env.step(action_id)

        r, c = self.agent_location
        close_to_fire_wall = False
        # NORTH
        if (0 <= (r - 1) <= self.map_size - 1) and self.map[r - 1][c] == self.env.items_id['fire_wall']:
            close_to_fire_wall = True
        # SOUTH
        elif (0 <= (r + 1) <= self.map_size - 1) and self.map[r + 1][c] == self.env.items_id['fire_wall']:
            close_to_fire_wall = True
        # WEST
        elif (0 <= (c - 1) <= self.map_size - 1) and self.map[r][c - 1] == self.env.items_id['fire_wall']:
            close_to_fire_wall = True
        # EAST
        elif (0 <= (c + 1) <= self.map_size - 1) and self.map[r][c + 1] == self.env.items_id['fire_wall']:
            close_to_fire_wall = True

        if close_to_fire_wall:
            reward = -self.reward_done // 2
            done = True
            info['message'] = 'You died due to fire_wall'

        # Update after each step
        self.env.last_reward = reward
        self.env.last_done = done

        lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                 'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                 'last_done': self.env.last_done}
        self.set_lasts(lasts)

        return obs, reward, done, info


def remap_action_difficulty(env, difficulty='hard'):
    """
    Remap actions randomly
    If LimitActions wrapper is used only limited_actions will be remapped regardless of any difficulty
    """

    if hasattr(env, 'limited_actions_id'):
        env.set_limited_actions_id(env.remap_action(env.limited_actions_id, 0))
    else:
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
        self.env.items_id.setdefault('fence', len(self.items_id))

    def step(self, action_id):

        old_rubber_quantity = copy.deepcopy(self.env.inventory_items_quantity['rubber'])

        obs, reward, done, info = self.env.step(action_id)

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

        return obs, reward, done, info


class AddChopAction(gym.core.Wrapper):
    """
    Novelty wrapper to add chop action
    It's like break action, but instead of 1 item, agent will get 2 items, but step_cost will be higher (1.2 times)
    """

    def __init__(self, env):
        super().__init__(env)

        self.env.manipulation_actions_id['Chop'] = len(self.actions_id)
        self.env.actions_id.update(self.manipulation_actions_id)
        self.action_space = spaces.Discrete(len(self.actions_id))

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Chop' in self.limited_actions_id, "Cannot use addchop novelty because you do not have Chop in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Chop']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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

                reward = self.reward_intermediate

            else:
                result = False
                message = "Cannot chop " + self.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class AddJumpAction(gym.core.Wrapper):
    """
    Novelty wrapper to add jump action, when it's executed, the agent jumps 2 blocks forward
    """

    def __init__(self, env):
        super().__init__(env)

        self.env.manipulation_actions_id['Jump'] = len(self.actions_id)
        self.env.actions_id.update(self.manipulation_actions_id)
        self.action_space = spaces.Discrete(len(self.actions_id))

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Jump' in self.limited_actions_id, "Cannot use addjump novelty because you do not have Jump in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Jump']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

            r, c = self.agent_location

            reward = -1  # default reward
            result = True
            step_cost = 0  # default step_cost
            message = ''

            if self.agent_facing_str == 'NORTH' and (0 <= (r - 2) <= self.map_size - 1) and self.map[r - 2][c] == 0:
                self.set_agent_location(r - 2, c)
            elif self.agent_facing_str == 'SOUTH' and (0 <= (r + 2) <= self.map_size - 1) and self.map[r + 2][c] == 0:
                self.set_agent_location(r + 2, c)
            elif self.agent_facing_str == 'WEST' and (0 <= (c - 2) <= self.map_size - 1) and self.map[r][c - 2] == 0:
                self.set_agent_location(r, c - 2)
            elif self.agent_facing_str == 'EAST' and (0 <= (c + 2) <= self.map_size - 1) and self.map[r][c + 2] == 0:
                self.set_agent_location(r, c + 2)
            else:
                result = False
                message = 'Block in path'

            step_cost = 27.906975 * 2

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class BreakIncrease(gym.core.Wrapper):
    """
    Novelty wrapper to get 2 items in inventory when the agent break that item instead of 1
    itemtobreakmore: apply this to only itemtobreakmore or if itemtobreakmore == '', then to all items
    """

    def __init__(self, env, itemtobreakmore=''):
        super().__init__(env)

        self.itemtobreakmore = itemtobreakmore

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            assert 'Break' in self.limited_actions_id, "Cannot use breakincrease novelty because you do not have Break in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        if action_id == actions_id['Break']:
            self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

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

                reward = self.reward_intermediate

            else:
                result = False
                message = "Cannot break " + self.env.block_in_front_str

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


class ExtractIncDec(gym.core.Wrapper):
    """
    Novelty wrapper to increase/decrease string when Extract_string action is executed
    incdec: 'increase' to increase, 'decrease' to decrease
    """

    def __init__(self, env, incdec='decrease'):
        super().__init__(env)

        self.incdec = incdec

    def step(self, action_id):

        if hasattr(self, 'limited_actions_id'):
            has_extract = False
            for action in self.limited_actions_id:
                if action.startswith('Extract'):
                    has_extract = True
                    break
            assert has_extract, "Cannot use extractincdec novelty because you do not have Extract action in LimitActions"
            actions_id = self.limited_actions_id
        else:
            actions_id = self.actions_id

        self.env.last_action = list(actions_id.keys())[list(actions_id.values()).index(action_id)]

        if self.env.last_action.startswith('Extract'):

            reward = -1  # default reward
            result = True
            step_cost = 120.0  # default step_cost
            message = ''

            if self.env_id.startswith('NovelGridworld-Bow'):
                if self.block_in_front_str == 'wool':
                    if self.incdec == 'increase':
                        self.inventory_items_quantity['string'] += 4 * 2  # Extract_string
                    else:
                        self.inventory_items_quantity['string'] += 4 // 2  # Extract_string
                    block_r, block_c = self.block_in_front_location
                    self.map[block_r][block_c] = 0
                    reward = self.reward_intermediate
                    step_cost = 5000
                else:
                    result = False
                    message = "No wool found"
            elif self.env_id.startswith('NovelGridworld-Pogostick'):
                # Make sure that block_in_front_location is next to a tree
                block_in_front_next_to_tree = self.is_block_in_front_next_to('tree_log')
                if self.block_in_front_str == 'tree_tap':
                    if block_in_front_next_to_tree:
                        if self.incdec == 'increase':
                            self.inventory_items_quantity['rubber'] += 1 * 2  # Extract_rubber
                        reward = self.reward_intermediate
                        step_cost = 50000
                    else:
                        result = False
                        message = "No tree_log near tree_tap"
                else:
                    result = False
                    message = "No tree_tap found"

            # Update after each step
            self.env.grab_entities()
            if hasattr(self, 'observation'):
                obs = self.observation()
            else:
                obs = self.env.get_observation()
            self.env.update_block_in_front()

            done = False
            if self.env.inventory_items_quantity[self.goal_item_to_craft] >= 1:
                reward = self.reward_done
                done = True

            info = {'result': result, 'step_cost': step_cost, 'message': message}

            # Update after each step
            self.env.step_count += 1
            self.env.last_step_cost = step_cost
            self.env.last_reward = reward
            self.env.last_done = done

            lasts = {'last_action': self.env.last_action, 'step_count': self.env.step_count,
                     'last_step_cost': self.env.last_step_cost, 'last_reward': self.env.last_reward,
                     'last_done': self.env.last_done}
            self.set_lasts(lasts)
        else:
            obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info


#################### Novelty Helper ####################

def inject_novelty(env, novelty_name, difficulty='hard', novelty_arg1='', novelty_arg2=''):

    novelty_names = ['addchop', 'additem', 'addjump', 'axe', 'axetobreak', 'breakincrease', 'crate', 'extractincdec', 'fence',
                     'fencerestriction', 'firewall', 'remapaction', 'replaceitem', 'coincraft']
    assert novelty_name in novelty_names, "novelty_name must be one of " + str(novelty_names)
    if novelty_name in ['additem', 'axe', 'axetobreak', 'crate', 'fence', 'fencerestriction', 'firewall', 'remapaction', 'replaceitem']:
        assert difficulty in ['easy', 'medium', 'hard'], "difficulty must be one of 'easy', 'medium', 'hard'"

    if novelty_name == 'addchop':
        env = AddChopAction(env)
    elif novelty_name == 'additem':
        assert novelty_arg1, "For additem novelty, novelty_arg1 (name of the item to add) is needed"

        env = AddItem(env, difficulty, novelty_arg1)
    elif novelty_name == 'addjump':
        env = AddJumpAction(env)
    elif novelty_name == 'axe':
        assert novelty_arg1 in ['wooden', 'iron'], \
            "For axe novelty, novelty_arg1 (attribute of axe, e.g. wooden, iron) is needed"

        if novelty_arg2:
            assert novelty_arg2 in ['true', 'false'], "For axe novelty, novelty_arg2 (breakincrease) must be 'true' or 'false'"

            if difficulty == 'easy':
                env = AxeEasy(env, novelty_arg1, novelty_arg2)
            elif difficulty == 'medium':
                env = AxeMedium(env, novelty_arg1, novelty_arg2)
            elif difficulty == 'hard':
                env = AxeHard(env, novelty_arg1, novelty_arg2)
        else:
            if difficulty == 'easy':
                env = AxeEasy(env, novelty_arg1)
            elif difficulty == 'medium':
                env = AxeMedium(env, novelty_arg1)
            elif difficulty == 'hard':
                env = AxeHard(env, novelty_arg1)
    elif novelty_name == 'axetobreak':
        assert novelty_arg1 in ['wooden', 'iron'], \
            "For axe novelty, novelty_arg1 (attribute of axe, e.g. wooden, iron) is needed"

        if difficulty == 'easy':
            env = AxetoBreakEasy(env, novelty_arg1)
        elif difficulty == 'medium':
            env = AxetoBreakMedium(env, novelty_arg1)
        elif difficulty == 'hard':
            env = AxetoBreakHard(env, novelty_arg1)
    elif novelty_name == 'breakincrease':
        if novelty_arg1:
            assert novelty_arg1 in env.items, env.itemtobreakmore + " is not in " + env.env_id

            env = BreakIncrease(env, novelty_arg1)
        else:
            env = BreakIncrease(env)
    elif novelty_name == 'crate':
        env = Crate(env, difficulty)
    elif novelty_name == 'extractincdec':
        assert novelty_arg1 in ['increase', 'decrease'], \
            "For extractincdec novelty, novelty_arg1 ('increase', 'decrease') is needed"

        assert env.env_id != 'NovelGridworld-Bow-v0', "There is nothing to extract in NovelGridworld-Bow-v0"

        if env.env_id == 'NovelGridworld-Bow-v1':
            assert novelty_arg1 == 'decrease', "In NovelGridworld-Bow-v1, increasing string extraction will not benefit " \
                                               "as only 3 string are needed"

        assert not env.env_id.startswith('NovelGridworld-Pogostick'), "In NovelGridworld-Pogostick, you should not use " \
            "extractincdec novelty because rubber extraction cannot be decreased, and increasing rubber extraction will" \
            " not benefit as only 1 rubber is needed"

        env = ExtractIncDec(env, novelty_arg1)
    elif novelty_name == 'fence':
        assert novelty_arg1, "For fence novelty, novelty_arg1 (attribute of fence, e.g. oak, jungle) is needed"

        env = Fence(env, difficulty, novelty_arg1)
    elif novelty_name == 'fencerestriction':
        assert novelty_arg1, "For fencerestriction novelty, novelty_arg1 (attribute of fence, e.g. oak, jungle) is needed"

        env = FenceRestriction(env, difficulty, novelty_arg1)
    elif novelty_name == 'firewall':
        env = FireWall(env, difficulty)
    elif novelty_name == 'remapaction':
        env = remap_action_difficulty(env, difficulty)
    elif novelty_name == 'replaceitem':
        assert novelty_arg1 and novelty_arg2, "For replaceitem novelty, novelty_arg1 (Item to replace) and novelty_arg2" \
                                              "(Item to replace with) are needed"

        env = ReplaceItem(env, difficulty, novelty_arg1, novelty_arg2)
    elif novelty_name == 'coincraft':
        env = CoinCraft(env, 'medium')

    return env
