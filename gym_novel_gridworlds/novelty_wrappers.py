import copy
import time
import numpy as np

import gym
from gym import error, spaces, utils


class AxetoBreakEasy(gym.core.Wrapper):
    """
    Novelty wrapper to add a new item (axe) in the inventory and requiring axe to break items
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env):
        super().__init__(env)

        self.axe_name = 'wooden_axe'  # wooden_axe, iron_axe
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.items_lidar.append(self.axe_name)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)        
        self.action_space = spaces.Discrete(len(self.env.actions_id))
        self.env.low = np.array([0] * (len(self.env.items_lidar) * self.env.num_beams) + [0] * len(self.env.inventory_items_quantity) + [0])
        self.env.high = np.array([self.env.max_beam_range] * (len(self.env.items_lidar) * self.env.num_beams) + [10] * len(
            self.env.inventory_items_quantity) + [10])  # maximum 10 of an object present in the env, and selected item's id is passed. Need to one hot encode it        
        self.observation_space = spaces.Box(self.env.low, self.env.high, dtype=int)

    def reset(self, reset_from_failed_state = False, env_instance = None):
        # Modified the reset function to take the arguments for resetting to the failed state. 
        obs = self.env.reset(reset_from_failed_state = reset_from_failed_state, env_instance = env_instance)

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
                    self.items_quantity[self.env.block_in_front_str] -= 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1
                    self.items_quantity[self.env.block_in_front_str] -= 1

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
    Novelty wrapper to add a new item (axe) in the map
    When the agent goes near axe, axe gets into the inventory
    Axe is required to break items
    Using axe reduces the step_cost when Break action is used
    """

    def __init__(self, env):
        super().__init__(env)

        self.axe_name = 'wooden_axe'  # wooden_axe, iron_axe
        self.env.add_new_items({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.items_quantity_at_start.update({self.axe_name:1}) # need to update the novel item at the start
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.hierarchical_actions.update({'Approach ' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.hierarchical_actions)
        self.env.items_lidar.append(self.axe_name)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)
        self.action_space = spaces.Discrete(len(self.env.actions_id))
        self.env.low = np.array([0] * (len(self.env.items_lidar) * self.env.num_beams) + [0] * len(self.env.inventory_items_quantity) + [0])
        self.env.high = np.array([self.env.max_beam_range] * (len(self.env.items_lidar) * self.env.num_beams) + [10] * len(
            self.env.inventory_items_quantity) + [10])  # maximum 10 of an object present in the env, and selected item's id is passed. Need to one hot encode it        
        self.observation_space = spaces.Box(self.env.low, self.env.high, dtype=int)        


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
                    self.items_quantity[self.env.block_in_front_str] -= 1

                    reward = self.reward_intermediate

                    step_cost = step_cost * 0.5  # 1800.0
                elif self.env.inventory_items_quantity[self.axe_name] >= 1 and self.env.selected_item == 'iron_axe':
                    block_r, block_c = self.env.block_in_front_location
                    self.env.map[block_r][block_c] = 0
                    self.env.inventory_items_quantity[self.env.block_in_front_str] += 1
                    self.items_quantity[self.env.block_in_front_str] -= 1

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

class FireCraftingTableEasy(gym.core.Wrapper):
    '''
    Novelty wrapper to set crafting table on fire. Add a new object called water in agent's 
    Inventory and add a new action called "spray". The agent needs to spray water on the crafting table
    to access it.
    '''
    def __init__(self, env):
        super().__init__(env)

        self.water_name = 'water'  # wooden_axe, iron_axe
        self.env.items.add(self.water_name)
        self.env.items_id.setdefault(self.water_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.water_name: 1})
        self.env.entities.add(self.water_name)
        self.env.select_actions_id.update({'Select_' + self.water_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.actions_id.update({'Spray':len(self.env.actions_id)})
        # print("\n actions ID are: ", self.env.actions_id)
        self.env.items_lidar.append(self.water_name)
        # print("items lidar: ", self.env.items_lidar)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)
        self.action_space = spaces.Discrete(len(self.env.actions_id))
        self.env.low = np.array([0] * (len(self.env.items_lidar) * self.env.num_beams) + [0] * len(self.env.inventory_items_quantity) + [0]+[0])
        self.env.high = np.array([self.env.max_beam_range] * (len(self.env.items_lidar) * self.env.num_beams) + [10] * len(
            self.env.inventory_items_quantity) + [10] + [2])  # maximum 10 of an object present in the env, and selected item's id is passed. Need to one hot encode it        
        self.observation_space = spaces.Box(self.env.low, self.env.high, dtype=int)     
        # print("Observation space dim: ", self.observation_space.shape[0])
        self.is_crafting_table_on_fire = True   

    def reset(self, reset_from_failed_state = False, env_instance = None):
        # Modified the reset function to take the arguments for resetting to the failed state. 
        obs = self.env.reset(reset_from_failed_state = reset_from_failed_state, env_instance = env_instance)
        # obs = np.append(obs,1) # 0 indicates the CT is on Fire
        self.is_crafting_table_on_fire = True   
        self.env.inventory_items_quantity.update({self.water_name: 1})

        return obs

    def get_observation(self):
        obs = self.env.get_observation()
        if self.is_crafting_table_on_fire == False:
            obs = np.append(obs,0)
        else:
            obs = np.append(obs,1)
        return obs

    def step(self, action_id):

        if action_id == self.actions_id['Craft_tree_tap'] or action_id == self.actions_id['Craft_pogo_stick']:
            if self.is_crafting_table_on_fire == False:
                obs, reward, done, info = self.env.step(action_id)  
                obs = self.get_observation()
                return obs, reward, done, info
            else:
                info = self.env.get_info()
                info['result'] = False
                obs = self.get_observation()
                return obs, -1, False, info
        elif action_id == self.actions_id['Spray']:
            self.env.update_block_in_front()
            if self.env.selected_item == self.water_name and self.env.block_in_front_str == 'crafting_table':
                self.is_crafting_table_on_fire = False
            obs = self.get_observation()
            info = self.env.get_info()
            info['result'] = True
            return obs, -1, False, info
        else:
            obs, reward, done, info = self.env.step(action_id)
            obs = self.get_observation()
            return obs, reward, done, info

class FireCraftingTableHard(gym.core.Wrapper):
    '''
    Novelty wrapper to set crafting table on fire. Add a new object called water in agent's 
    environment and add a new action called "spray". The agent needs to spray water on the crafting table
    to access it.
    '''
    def __init__(self, env):
        super().__init__(env)

        self.water_name = 'water' 

        self.env.add_new_items({self.water_name: 1})
        self.env.entities.add(self.water_name)
        self.env.items_lidar.append(self.water_name)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)
        self.env.items_quantity_at_start.update({self.water_name:1})
        # self.env.inventory_items_quantity.update({self.water_name: 1})
        self.env.select_actions_id.update({'Select_' + self.water_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.actions_id.update({'Spray':len(self.env.actions_id)})
        self.env.hierarchical_actions.update({'Approach ' + self.water_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.hierarchical_actions)        
        # print("\n actions ID are: ", self.env.actions_id)
        self.action_space = spaces.Discrete(len(self.env.actions_id))
        self.env.low = np.array([0] * (len(self.env.items_lidar) * self.env.num_beams) + [0] * len(self.env.inventory_items_quantity) + [0] + [0])
        self.env.high = np.array([self.env.max_beam_range] * (len(self.env.items_lidar) * self.env.num_beams) + [30] * len(
            self.env.inventory_items_quantity) + [15] + [0])  # maximum 10 of an object present in the env, and selected item's id is passed. Need to one hot encode it        
        self.observation_space = spaces.Box(self.env.low, self.env.high, dtype=int)     
        
        self.is_crafting_table_on_fire = True   

    def reset(self, reset_from_failed_state = False, env_instance = None):
        # Modified the reset function to take the arguments for resetting to the failed state. 
        obs = self.env.reset(reset_from_failed_state = reset_from_failed_state, env_instance = env_instance)
        self.is_crafting_table_on_fire = True   
        obs = np.append(obs,1) # 0 indicates the CT is on Fire
        return obs

    def get_observation(self):
        obs = self.env.get_observation()
        if self.is_crafting_table_on_fire == False:
            obs = np.append(obs,0)
        else:
            obs = np.append(obs,1)
        return obs

    def step(self, action_id):
        if action_id == self.actions_id['Craft_tree_tap'] or action_id == self.actions_id['Craft_pogo_stick']:
            if self.is_crafting_table_on_fire == False:
                self.env.update_block_in_front()
                obs, reward, done, info = self.env.step(action_id)
                obs = self.get_observation()
                return obs, reward, done, info
            else:
                info = self.env.get_info()
                info['result'] = False
                obs = self.get_observation()
                return obs, -1, False, info
        elif action_id == self.actions_id['Spray']:
            self.env.update_block_in_front()
            reward = -1
            if self.env.selected_item == self.water_name and self.env.block_in_front_str == 'crafting_table':
                self.is_crafting_table_on_fire = False
            info = self.env.get_info()
            info['result'] = True
            obs = self.get_observation()
            return obs, reward, False, info
        else:
            obs, reward, done, info = self.env.step(action_id)
            obs = self.get_observation()
            return obs, reward, done, info


class RubberTree(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
        self.rubber_tree_name = 'rubber_tree'  # wooden_axe, iron_axe
        self.env.items.add(self.rubber_tree_name)
        self.env.items_quantity.update({self.rubber_tree_name:1})
        self.env.items_quantity_at_start.update({self.rubber_tree_name:1})
        self.env.items_id.setdefault(self.rubber_tree_name, len(self.items_id))
        self.env.unbreakable_items.add(self.rubber_tree_name)
        self.env.items_lidar.append(self.rubber_tree_name)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)
        self.env.unselectable_items.add(self.rubber_tree_name)
        self.env.hierarchical_actions.update({'Approach ' + self.rubber_tree_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.hierarchical_actions)                        
        self.env.inventory_items_quantity.update({self.rubber_tree_name:0})
        self.env.low = np.array([0] * (len(self.env.items_lidar) * self.env.num_beams) + [0] * len(self.env.inventory_items_quantity) + [0])
        self.env.high = np.array([self.env.max_beam_range] * (len(self.env.items_lidar) * self.env.num_beams) + [10] * len(
            self.env.inventory_items_quantity) + [10])  # maximum 10 of an object present in the env, and selected item's id is passed. Need to one hot encode it        
        self.observation_space = spaces.Box(self.env.low, self.env.high, dtype=int)     

    def step(self, action_id):

        if action_id == self.actions_id['Extract_rubber']:
            self.update_block_in_front()
            if self.block_in_front_str == self.rubber_tree_name and self.selected_item == 'tree_tap':
                self.inventory_items_quantity['rubber'] += 1  # Extract_rubber
                reward = -1
                info = self.env.get_info()
                info['result'] = True
                return self.env.get_observation(), reward, False, info
            else:
                info = self.env.get_info()
                info['result'] = False
                return self.env.get_observation(), -1, False, info 
        else:
            obs, reward, done, info = self.env.step(action_id)
            return obs, reward, done, info

class AxeBreakFireCTEasy(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.water_name = 'water'  
        self.env.items.add(self.water_name)
        self.env.items_id.setdefault(self.water_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.water_name: 1})
        self.env.entities.add(self.water_name)
        self.env.items_lidar.append(self.water_name)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)
        self.is_crafting_table_on_fire = True       

        self.axe_name = 'wooden_axe' 
        self.env.items.add(self.axe_name)
        self.env.items_id.setdefault(self.axe_name, len(self.items_id))
        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.env.entities.add(self.axe_name)
        self.env.items_lidar.append(self.axe_name)
        self.env.items_id_lidar = self.env.set_items_id(self.env.items_lidar)        

        self.env.select_actions_id.update({'Select_' + self.water_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.env.actions_id.update({'Spray':len(self.env.actions_id)})
        self.env.select_actions_id.update({'Select_' + self.axe_name: len(self.env.actions_id)})
        self.env.actions_id.update(self.env.select_actions_id)
        self.action_space = spaces.Discrete(len(self.env.actions_id))

        self.env.low = np.array([0] * (len(self.env.items_lidar) * self.env.num_beams) + [0] * len(self.env.inventory_items_quantity) + [0] + [0])
        self.env.high = np.array([self.env.max_beam_range] * (len(self.env.items_lidar) * self.env.num_beams) + [10] * len(
            self.env.inventory_items_quantity) + [10] + [2])  # maximum 10 of an object present in the env, and selected item's id is passed. Need to one hot encode it        
        self.observation_space = spaces.Box(self.env.low, self.env.high, dtype=int)     

    def reset(self, reset_from_failed_state = False, env_instance = None):
        # Modified the reset function to take the arguments for resetting to the failed state. 
        obs = self.env.reset(reset_from_failed_state = reset_from_failed_state, env_instance = env_instance)

        self.env.inventory_items_quantity.update({self.axe_name: 1})
        self.is_crafting_table_on_fire = True   
        self.env.inventory_items_quantity.update({self.water_name: 1})
        obs = np.append(obs,1) # 0 indicates the CT is on Fire
        return obs

    def get_observation(self):
        obs = self.env.get_observation()
        if self.is_crafting_table_on_fire == False:
            obs = np.append(obs,0)
        else:
            obs = np.append(obs,1)        
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
                    self.items_quantity[self.env.block_in_front_str] -= 1

                    reward = self.reward_intermediate
                    step_cost = step_cost * 0.5  # 1800.0
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
            return obs, reward, done, info
        
        elif action_id == self.actions_id['Craft_tree_tap'] or action_id == self.actions_id['Craft_pogo_stick']:
            if self.is_crafting_table_on_fire == False:
                obs, reward, done, info = self.env.step(action_id)
                obs = self.get_observation()
                return obs, reward, done, info
            else:
                info = self.env.get_info()
                info['result'] = False
                obs = self.get_observation()
                return obs, -1, False, info
        elif action_id == self.actions_id['Spray']:
            if self.env.selected_item == self.water_name and self.env.block_in_front_str == 'crafting_table':
                self.is_crafting_table_on_fire = False
            obs = self.get_observation()
            info = self.env.get_info()
            info['result'] = True
            return obs, -1, False, info
        else:
            obs, reward, done, info = self.env.step(action_id)
            obs = self.get_observation()                   
            return obs, reward, done, info

class ScrapePlank(gym.core.Wrapper):
    '''
    Novelty wrapper that fails the break action. The agent can no longer receive tree_log from the environment. 
    The agent now has to perform ScrapePlank while being in front of a tree_log to receive plank.
    The agent should be able to move forward in the plan.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.unbreakable_items.add('tree_log')
        self.env.actions_id.update({'Scrapeplank':len(self.env.actions_id)})
        self.action_space = spaces.Discrete(len(self.env.actions_id))
        
    def step(self, action_id):

        if action_id == self.actions_id['Scrapeplank']:
            if self.env.block_in_front_str == 'tree_log':
                block_r, block_c = self.block_in_front_location
                self.map[block_r][block_c] = 0
                self.inventory_items_quantity['plank'] += 4
                reward = -1
                info = self.env.get_info()
                info['result'] = True
                return self.env.get_observation(), reward, False, info     
            else:
                info = self.env.get_info()
                info['result'] = False                
                return self.env.get_observation(), -1, False, self.env.get_info()            
        else:
            obs, reward, done, info = self.env.step(action_id)
            return obs, reward, done, info

#################### Novelty Helper ####################

def inject_novelty(env, novelty_name):

    novelty_names = ['axetobreakeasy', 'axetobreakhard', 'firecraftingtableeasy','firecraftingtablehard', 'rubbertree', 'axefirecteasy', 'scrapeplank']
    assert novelty_name in novelty_names, "novelty_name must be one of " + str(novelty_names)

    if novelty_name == 'axetobreakeasy':
        env = AxetoBreakEasy(env)
    elif novelty_name == 'axetobreakhard':
        env = AxetoBreakHard(env)        
    elif novelty_name == 'firecraftingtableeasy':
        env = FireCraftingTableEasy(env)
    elif novelty_name == 'firecraftingtablehard':
        env = FireCraftingTableHard(env)        
    elif novelty_name == 'rubbertree':
        env = RubberTree(env)
    elif novelty_name == 'axefirecteasy':
        env = AxeBreakFireCTEasy(env)
    elif novelty_name == 'scrapeplank':
        env = ScrapePlank(env)
        
    return env
