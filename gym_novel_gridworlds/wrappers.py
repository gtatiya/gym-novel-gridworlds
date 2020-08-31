import pickle
import os
from datetime import datetime

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
                 "map": self.env.map,
                 "agent_location": self.env.agent_location,
                 "agent_facing_str": self.env.agent_facing_str,
                 "block_in_front_id": self.env.block_in_front_id,

                 "items_id": self.env.items_id,
                 "items_quantity": self.env.items_quantity,
                 "inventory_items_quantity": self.env.inventory_items_quantity,

                 "action_str": self.env.action_str,
                 "last_action": self.env.last_action,

                 "last_done": self.last_done}

        return state

    def save(self):

        path = os.path.join(self.save_path,
                            datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_{env}.bin".format(env=self.env.env_name))

        f = open(path, 'wb')
        pickle.dump(self.state_trajectories, f)
        f.close()
        print("Trajectories saved at: ", path)
