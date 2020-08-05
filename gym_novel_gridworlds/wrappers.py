import math
import pickle
import os
from datetime import datetime

import numpy as np

import gym
from gym import error, spaces, utils


class SaveTrajectories(gym.core.Wrapper):
    """
    Wrapper to save agent trajectories in the environment
    """

    def __init__(self, env, save_path):
        super().__init__(env)

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.state_trajectories = []

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        state = self.get_state()
        self.state_trajectories.append(state)

        return obs, reward, done, info

    def get_state(self):

        state = {"map_size": self.env.map_size,
                 "items_id": self.env.items_id,
                 "items_quantity": self.env.items_quantity,
                 "inventory_items_quantity": self.env.inventory_items_quantity,
                 "agent_location": self.env.agent_location,
                 "agent_facing_str": self.env.agent_facing_str,
                 "map": self.env.map,
                 "action_str": self.env.action_str,
                 "last_action": self.env.last_action,
                 "block_in_front_id": self.env.block_in_front_id}

        return state

    def save(self):

        path = os.path.join(self.save_path,
                            datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_{env}.bin".format(env=self.env.env_name))

        f = open(path, 'wb')
        pickle.dump(self.state_trajectories, f)
        f.close()
        print("Trajectories saved at: ", path)


class LidarInFront(gym.core.ObservationWrapper):
    """
    Send several beans (self.num_beams) at equally spaced angles in 180 degrees in front of agent
    """

    def __init__(self, env):
        super().__init__(env)

        # Observation Space
        self.num_beams = 5
        self.max_beam_range = int(math.sqrt(2 * (self.env.map_size - 2) ** 2))  # Hypotenuse of a square
        low = np.ones(len(self.env.items_id) * self.env.num_beams, dtype=int)
        high = np.array([self.env.max_beam_range] * len(self.env.items_id) * self.env.num_beams)
        self.observation_space = spaces.Box(low, high, dtype=int)

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

    def observation(self, obs):
        return self.get_lidarSignal()
