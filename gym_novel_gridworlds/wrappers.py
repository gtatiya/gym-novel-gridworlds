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

    def step(self, action_id):
        obs, reward, done, info = self.env.step(action_id)

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

                 "action_str": self.env.actions_id,
                 "last_action": self.env.last_action,

                 "last_done": self.last_done}

        return state

    def save(self):
        path = os.path.join(self.save_path,
                            datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_{env}.bin".format(env=self.env.env_id))

        f = open(path, 'wb')
        pickle.dump(self.state_trajectories, f)
        f.close()
        print("Trajectories saved at: ", path)


class LimitActions(gym.core.Wrapper):
    """
    Wrapper to limit the actions in the environment
    limited_actions: set of actions to use
    """

    def __init__(self, env, limited_actions):
        super().__init__(env)

        self.limited_actions = limited_actions
        self.limited_actions_id = {action: i for i, action in enumerate(sorted(self.limited_actions))}
        self.action_space = spaces.Discrete(len(self.limited_actions_id))

    def step(self, action_id):

        assert action_id in self.limited_actions_id.values(), "Action ID " + str(action_id) + " is not valid, max" \
                                                              "action ID is " + str(len(self.limited_actions_id)-1)
        last_action = list(self.limited_actions_id.keys())[list(self.limited_actions_id.values()).index(action_id)]

        assert last_action in self.actions_id, last_action + " is not a valid action for " + self.env_id
        action_id = self.actions_id[last_action]

        obs, reward, done, info = self.env.step(action_id)

        return obs, reward, done, info
