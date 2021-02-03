import math
import copy

import numpy as np

import gym
from gym import error, spaces, utils


class LidarInFront(gym.core.ObservationWrapper):
    """
    Send several beans (self.num_beams) at equally spaced angles in 180 degrees in front of agent + agent's current
    inventory
    """

    def __init__(self, env, num_beams=5):
        super().__init__(env)

        # Observation Space
        items_to_exclude = ['air', self.goal_item_to_craft]       
        self.lidar_items = copy.deepcopy(self.env.items_id)
        list(map(self.lidar_items.pop, items_to_exclude))  # remove air and goal_item_to_craft from the lidar_items
        self.lidar_items_id = self.set_items_id(self.lidar_items)  # set IDs for all the lidar items
        self.num_beams = num_beams
        self.max_beam_range = int(math.sqrt(2 * (self.env.map_size - 2) ** 2))  # Hypotenuse of a square
        low = np.zeros(len(self.lidar_items) * self.num_beams + len(self.inventory_items_quantity), dtype=int)
        high = np.array([self.max_beam_range] * len(self.lidar_items) * self.num_beams + [20] * len(self.inventory_items_quantity))  # 20 denotes the maximum quantity of any item in the inventory
        self.observation_space = spaces.Box(low, high, dtype=int)

    def get_lidarSignal(self):
        """
        Send several beams (self.num_beams) at equally spaced angles in 360 degrees in front of agent within a range
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
            beam_signal = np.zeros(len(self.lidar_items_id), dtype=int)

            # Keep sending longer beams until hit an object or wall
            for beam_range in range(1, self.max_beam_range+1):
                r_obj = r + np.round(beam_range * x_ratio)
                c_obj = c + np.round(beam_range * y_ratio)
                obj_id_rc = self.map[int(r_obj)][int(c_obj)]

                # If bean hit an object or wall
                if obj_id_rc != 0:
                    item = list(self.items_id.keys())[list(self.items_id.values()).index(obj_id_rc)]
                    if item in self.lidar_items_id:
                        obj_id_rc = self.lidar_items_id[item]
                        beam_signal[obj_id_rc - 1] = beam_range
                    break

            lidar_signals.extend(beam_signal)

        return lidar_signals

    def observation(self, obs):
        """
        observation is lidarSignal + inventory_items_quantity
        :return: observation
        """

        lidar_signals = self.get_lidarSignal()
        observation = lidar_signals + list(self.inventory_items_quantity.values())
        
        return np.array(observation)


class AgentMap(gym.core.ObservationWrapper):
    """
    Agent's local view within a range (agent_view_size), agent_facing_id, inventory_items_quantity
    """

    def __init__(self, env):
        super().__init__(env)

        # Observation Space
        self.max_items = 20
        self.agent_view_size = 5

        assert not self.max_items < len(self.env.items), "Cannot have more than " + str(self.max_items) + " items"
        assert self.agent_view_size >= 1, "Increase the agent_view_size"

        self.observation_space = spaces.Box(low=0, high=self.max_items,
                                            shape=(self.agent_view_size, self.agent_view_size, 1))
        self.observation_space = spaces.Dict({'agent_map': self.observation_space})

    def get_agentView(self):
        """
        Slice map with 0 padding based on agent_view_size

        :return: local view of the agent
        """

        extend = [self.agent_view_size, self.agent_view_size]  # row and column
        pad_value = 0

        extend = np.asarray(extend)
        map_ext_shp = self.env.map.shape + 2 * np.array(extend)
        map_ext = np.full(map_ext_shp, pad_value)
        insert_idx = [slice(i, -i) for i in extend]
        map_ext[tuple(insert_idx)] = self.env.map

        region_idx = [slice(i, j) for i, j in zip(self.env.agent_location, extend * 2 + 1 + self.env.agent_location)]

        return map_ext[tuple(region_idx)]

    def observation(self, obs):

        observation = {'agent_map': self.get_agentView(),
                       'agent_facing_id': self.env.agent_facing_id,
                       'inventory_items_quantity': self.env.inventory_items_quantity
                       }

        return observation
