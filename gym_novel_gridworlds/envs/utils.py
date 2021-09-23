# from polycraft_tufts.rl_agent.dqn_lambda.detectors import get_create_success_func_from_predicate_set, get_create_success_func_from_failed_operator, get_inv_quant, get_world_quant, get_entity_quant
import numpy as np
import math
import matplotlib.pyplot as plt
import time

"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
show_animation = False


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                # print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show()

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]#,
                  # [-1, -1, math.sqrt(2)],
                  # [-1, 1, math.sqrt(2)],
                  # [1, -1, math.sqrt(2)],
                  # [1, 1, math.sqrt(2)]]

        return motion


# # Class to execute operators implemented as hardcoded sequences of actions for testing
# class HardcodedOperator:
#     def __init__(self, name, action_sequence, effect_set):
#         self.name = name
#         self.action_sequence = action_sequence
#         self.effect_set = effect_set
#         self.create_check_success_func = get_create_success_func_from_predicate_set(effect_set)
#         self.check_success_func = None
#         self.action_step = 0

#     def reset(self, obs, info, env):
#         self.check_success_func = self.create_check_success_func(obs, info)
#         self.action_step = 0
#         return True

#     def get_action(self):
#         if self.name == 'break minecraft:log' or self.name == 'extractRubber':
#             time.sleep(1)
#         # Returning none is indication that sequence is done
#         if self.action_step >= len(self.action_sequence):
#             return None
#         action = self.action_sequence[self.action_step]
#         self.action_step += 1
#         return action

#     def check_success(self, obs, info):
#         return self.check_success_func(obs, info)

# #TODO: Can just use planner for this?
# class CraftingOperator:
#     def __init__(self, name, craft_goal, effect_set):
#         self.name = name
#         self.craft_goal = craft_goal
#         self.plan = []
#         self.effect_set = effect_set
#         self.create_check_success_func = get_create_success_func_from_predicate_set(effect_set)
#         self.check_success_func = None
#         self.action_step = 0

#     def reset(self, obs, info, env):
#         self.plan = []
#         self.check_success_func = self.create_check_success_func(obs, info)
#         self.action_step = 0
#         self.actions_id = env.actions_id

#         #only have crafting operators for tap and pogostick now
#         if get_inv_quant(info, self.craft_goal) > 0:
#             return True

#         if info['block_in_front']['name']  != 'minecraft:crafting_table':
#             if get_inv_quant(info, 'minecraft:crafting_table') < 1:
#                 return False
#             else:
#                 if info['block_in_front']['name'] == 'minecraft:air':
#                     self.plan.append('PLACE_CRAFTING_TABLE')
#                 else:
#                     #check spaces around agent to see if any free
#                     posx = self.env.player['pos'][0]
#                     posy = self.env.player['pos'][2]
#                     orient = self.env.player['facing']
#                     dir_vals = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}
#                     start_val = dir_vals[orient]
#                     num_rots = None
#                     for goal_orient in dir_vals.keys():
#                         #Why was this conditional here?
#                         # if orient == goal_orient:
#                         #     continue
#                         # else:
#                         if goal_orient == 'NORTH':
#                             if self.env.map_to_plot[posx][posy - 1] == self.env.items_id['minecraft:air']:
#                                 goal_val = dir_vals[goal_orient]
#                                 num_rots = goal_val - start_val
#                                 break
#                         if goal_orient == 'SOUTH':
#                             if self.env.map_to_plot[posx][posy + 1] == self.env.items_id['minecraft:air']:
#                                 goal_val = dir_vals[goal_orient]
#                                 num_rots = goal_val - start_val
#                                 break
#                         if goal_orient == 'EAST':
#                             if self.env.map_to_plot[posx+1][posy] == self.env.items_id['minecraft:air']:
#                                 goal_val = dir_vals[goal_orient]
#                                 num_rots = goal_val - start_val
#                                 break
#                         if goal_orient == 'WEST':
#                             if self.env.map_to_plot[posx-1][posy] == self.env.items_id['minecraft:air']:
#                                 goal_val = dir_vals[goal_orient]
#                                 num_rots = goal_val - start_val
#                                 break
#                     if num_rots == 1 or num_rots == -3:
#                         self.plan.append('TURN 90')
#                     elif num_rots == 2 or num_rots == -2:
#                         self.plan.append('TURN 90')
#                         self.plan.append('TURN 90')
#                     elif num_rots == 3 or num_rots == -1:
#                         self.plan.append('TURN -90')
#                     elif num_rots is None:
#                         return False
#                     self.plan.append('PLACE_CRAFTING_TABLE')

#         num_pogostick_crafts = 0
#         num_tree_tap_crafts = 0
#         num_stick_crafts = 0
#         num_plank_crafts = 0

#         num_sticks = get_inv_quant(info, 'minecraft:stick')
#         num_planks = get_inv_quant(info, 'minecraft:planks')

#         #Assuming we only want one (or should we assume we want one more?)
#         if self.craft_goal == 'polycraft:tree_tap':
#             # num_tree_tap_crafts = 1 if get_inv_quant(info, 'polycraft:tree_tap') == 0 else 0
#             num_tree_tap_crafts = 1
#             # num_stick_crafts = ((1 - get_inv_quant(info, 'minecraft:sticks')) // 4) + 1
#             num_stick_crafts = 0 if num_sticks > 0 else 1 #(num_sticks // 4) + 1
#             # num_plank_crafts = ((5 - get_inv_quant(info, 'minecraft:planks')) // 4) + 1
#             # num_plank_crafts = 0 if num_planks > 4 else ((4 - get_inv_quant(info, 'minecraft:planks')) // 4) + 1
#             plank_thresh = 4 if num_stick_crafts == 1 else 6
#             num_plank_crafts = 0 if num_planks > 6 else ((6 - get_inv_quant(info, 'minecraft:planks')) // 4) + 1
#         elif self.craft_goal == 'polycraft:wooden_pogo_stick':
#             num_pogostick_crafts = 1
#             num_stick_crafts = 0 if num_sticks > 3 else 1
#             plank_thresh = 2 if num_stick_crafts == 1 else 0
#             num_plank_crafts = 0 if num_planks > plank_thresh else 1
#             # num_plank_crafts = 0 if num_planks > 0 else 1

#             # num_stick_crafts = ((3 - get_inv_quant(info, 'minecraft:stick')) // 4) + 1
#             # num_plank_crafts = ((1 - get_inv_quant(info, 'minecraft:planks')) // 4) + 1

#         for i in range(num_plank_crafts):
#             self.plan.append('CRAFT minecraft:planks')
#         for i in range(num_stick_crafts):
#             self.plan.append('CRAFT minecraft:stick')
#         for i in range(num_tree_tap_crafts):
#             self.plan.append('CRAFT polycraft:tree_tap')
#         for i in range(num_pogostick_crafts):
#             self.plan.append('CRAFT polycraft:wooden_pogo_stick')
#         return True

#     def get_action(self):
#         # Returning none is indication that sequence is done
#         if self.action_step >= len(self.plan):
#             return None
#         action = self.plan[self.action_step]
#         self.action_step += 1
#         return self.actions_id[action]

#     def check_success(self, obs, info):
#         return self.check_success_func(obs, info)

class AStarOperator:
    def __init__(self, name, goal_type, effect_set):
        self.name = name
        self.goal_type = goal_type
        self.plan = []
        self.action_step = 0
        self.effect_set = effect_set
        # self.create_check_success_func = get_create_success_func_from_predicate_set(effect_set)
        self.check_success_func = None

    def reset(self, obs, info, env):
        self.plan = []
        self.actions_id = env.actions_id
        self.action_step = 0
        self.check_success_func = self.create_check_success_func(obs, info)

        if info['block_in_front']['name'] == self.goal_type:
            return True

        #If moveTo crafting table, first try just putting it down if it's in the inventory
        # TODO: what if place_crafting_table stops working?
        #   then this op would fail, which is fine, would still plan to placed ones
        #       actually would count as execution error which is annoying
        if self.goal_type == 'minecraft:crafting_table':
            if get_inv_quant(info, 'minecraft:crafting_table') >= 1:
                if info['block_in_front']['name'] == 'minecraft:air':
                    self.plan.append('PLACE_CRAFTING_TABLE')
                    return True
                else:
                    # check spaces around agent to see if any free
                    posx = env.player['pos'][0]
                    posy = env.player['pos'][2]
                    orient = env.player['facing']
                    dir_vals = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}
                    start_val = dir_vals[orient]
                    num_rots = None
                    for goal_orient in dir_vals.keys():
                        # if orient == goal_orient:
                        #     continue
                        # else:
                        if goal_orient == 'NORTH':
                            if env.map_to_plot[posx][posy - 1] == env.items_id['minecraft:air']:
                                goal_val = dir_vals[goal_orient]
                                num_rots = goal_val - start_val
                                break
                        if goal_orient == 'SOUTH':
                            if env.map_to_plot[posx][posy + 1] == env.items_id['minecraft:air']:
                                goal_val = dir_vals[goal_orient]
                                num_rots = goal_val - start_val
                                break
                        if goal_orient == 'EAST':
                            if env.map_to_plot[posx + 1][posy] == env.items_id['minecraft:air']:
                                goal_val = dir_vals[goal_orient]
                                num_rots = goal_val - start_val
                                break
                        if goal_orient == 'WEST':
                            if env.map_to_plot[posx - 1][posy] == env.items_id['minecraft:air']:
                                goal_val = dir_vals[goal_orient]
                                num_rots = goal_val - start_val
                                break
                    if num_rots is not None:
                        if num_rots == 1 or num_rots == -3:
                            self.plan.append('TURN 90')
                        elif num_rots == 2 or num_rots == -2:
                            self.plan.append('TURN 90')
                            self.plan.append('TURN 90')
                        elif num_rots == 3 or num_rots == -1:
                            self.plan.append('TURN -90')
                        self.plan.append('PLACE_CRAFTING_TABLE')
                        return True

        # Sample goal from items_locs
        # start position
        # x,z,y
        sx = env.player['pos'][0]
        sy = env.player['pos'][2]
        so = env.player['facing']

        grid_size = 1.0
        robot_radius = 0.9

        # obstacle positions
        ox, oy = [], []
        for r in range(len(env.binary_map[0])):
            for c in range(len(env.binary_map[1])):
                if env.binary_map[r][c] == 1:
                    ox.append(c)
                    oy.append(r)
        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
        success = False
        if get_world_quant(info, self.goal_type) > 0:
            # success = self.plan_to_random([self.goal_type], env.items_location, a_star, sx, sy, so)
            success = self.plan_to_nearest([self.goal_type], env.items_location, a_star, sx, sy, so)
        #if moveTo crafting_table and it's currently an entity in the world - pick it up and place it
        elif self.goal_type == 'minecraft:crafting_table' and get_entity_quant(info, self.goal_type) > 0:
            # success = self.plan_to_random([self.goal_type], env.entities_location, a_star, sx, sy, so)
            success = self.plan_to_nearest([self.goal_type], env.entities_location, a_star, sx, sy, so)
            #block you came from has to be open
            self.plan.append('TURN 90')
            self.plan.append('TURN 90')
            self.plan.append('PLACE_CRAFTING_TABLE')
        #only custom made moveTo operators are moveTo entity to pick up before rest of plan
        elif self.goal_type != 'minecraft:crafting_table' and self.goal_type != 'minecraft:log':
            success = self.plan_to_nearest([self.goal_type], env.entities_location, a_star, sx, sy, so)
            # success = self.plan_to_random([self.goal_type], env.entities_location, a_star, sx, sy, so)
            self.plan.append('MOVE w')

        return success
        # print(success, self.plan)

    #This operator will only be called in execution or reapplication of lost operator (e.g. in one instance couldn't
    #  find a path for moveTo but in another we could) so we should plan to nearest open
    #In reset_to_interesting_state we want random instance, but that won't be using this
    # def plan_to_random(self, interesting_items, items_location, a_star, sx, sy, so):
    def plan_to_nearest(self, interesting_items, items_location, a_star, sx, sy, so):
        # Then sample interesting blocks and go to them
        while len(interesting_items) != 0:
            # randomly sample item key of set to navigate towards (should mostly be len 1)
            interesting_item = interesting_items[np.random.randint(len(interesting_items))]
            try:
                interesting_item_locations = items_location[interesting_item].copy()
            except:
                interesting_item_locations = []


            #If few enough items, just iterate through and order all in terms of distance
            if len(interesting_item_locations) <= 10:
                interesting_item_dists = []
                for i in range(len(interesting_item_locations)):
                    interesting_instance = interesting_item_locations[i]
                    locs = interesting_instance.split(',')
                    dist = (sx - int(locs[0]))**2 + (sy - int(locs[2]))**2
                    interesting_item_dists.append(dist)
                while len(interesting_item_locations) != 0:
                    # randomly sample instance of item key to navigate towards
                    # ind = np.random.randint(len(interesting_item_locations))
                    #take nearest remaining instance
                    ind = np.argmin(interesting_item_dists)
                    interesting_instance = interesting_item_locations[ind]
                    locs = interesting_instance.split(',')
                    gx = int(locs[0])
                    gy = int(locs[2])
                    # Can't actually go into the item, so randomly sample point next to it to go to
                    relcoord = np.random.randint(4)
                    rx, ry = [], []
                    num_attempts = 0
                    while len(rx) < 2 and num_attempts < 4:
                        gx_ = gx
                        gy_ = gy
                        if relcoord == 0:
                            gx_ = gx + 1
                            ro = 'WEST'
                        elif relcoord == 1:
                            gx_ = gx - 1
                            ro = 'EAST'
                        elif relcoord == 2:
                            gy_ = gy + 1
                            ro = 'NORTH'
                        elif relcoord == 3:
                            gy_ = gy - 1
                            ro = 'SOUTH'
                        rx, ry = a_star.planning(sx, sy, gx_, gy_)
                        relcoord = (relcoord + 1) % 4
                        num_attempts += 1
                    if len(rx) > 1:
                        self.generateActionsFromPlan(sx, sy, so, rx, ry, ro)
                        # self.moveToUsingPlan(sy, sx, ry, rx)
                        return True
                    # if unreachable, delete location and keep trying
                    else:
                        del interesting_item_locations[ind]
                        del interesting_item_dists[ind]
            # #otherwise search out from agent and try one by one (don't want to get stuck on case where they spawn
            # # a bunch of instances
            else:
                print("TODO: implement spiral search for nearest goal instance, picking random")
                success = self.plan_to_random([interesting_item], items_location, a_star, sx, sy, so)
                if success:
                    return True
            interesting_items.remove(interesting_item)
        return False

    def plan_to_random(self, interesting_items, items_location, a_star, sx, sy, so):
        # Then sample interesting blocks and go to them
        while len(interesting_items) != 0:
            # randomly sample item key to navigate towards
            interesting_item = interesting_items[np.random.randint(len(interesting_items))]
            try:
                interesting_item_locations = items_location[interesting_item].copy()
            except:
                interesting_item_locations = []
            while len(interesting_item_locations) != 0:
                # randomly sample instance of item key to navigate towards
                ind = np.random.randint(len(interesting_item_locations))
                interesting_instance = interesting_item_locations[ind]
                locs = interesting_instance.split(',')
                gx = int(locs[0])
                gy = int(locs[2])
                # Can't actually go into the item, so randomly sample point next to it to go to
                # TODO: check if relcoord to item is vacant or reachable, otherwise we're wasting an opportunity
                relcoord = np.random.randint(4)
                rx, ry = [], []
                num_attempts = 0
                while len(rx) < 2 and num_attempts < 4:
                    gx_ = gx
                    gy_ = gy
                    if relcoord == 0:
                        gx_ = gx + 1
                        ro = 'WEST'
                    elif relcoord == 1:
                        gx_ = gx - 1
                        ro = 'EAST'
                    elif relcoord == 2:
                        gy_ = gy + 1
                        ro = 'NORTH'
                    elif relcoord == 3:
                        gy_ = gy - 1
                        ro = 'SOUTH'
                    rx, ry = a_star.planning(sx, sy, gx_, gy_)
                    relcoord = (relcoord + 1) % 4
                    num_attempts += 1
                # print(interesting_items)
                # print(interesting_item_locations)
                # print(interesting_item)
                # print(interesting_instance)
                # print('start', sx, sy, so)
                # print('finish', gx, gy, ro)
                # print('plan', rx, ry)
                # TODO: check if we ware already next to item are planning towards
                if len(rx) > 1:
                    self.generateActionsFromPlan(sx, sy, so, rx, ry, ro)
                    # self.moveToUsingPlan(sy, sx, ry, rx)
                    return True
                # if unreachable, delete location and keep trying
                else:
                    del interesting_item_locations[ind]
            interesting_items.remove(interesting_item)
        return False

    def generateActionsFromPlan(self, sx, sy, so, rxs, rys, ro):
        # sx, sy: start pos
        # rx, ry: subsequent locations to moveTo
        # rx, ry are backwards, iterate in reverse
        orientation = so
        for i in range(len(rxs) - 1):
            # First location is same as current location, skip
            ind = len(rxs) - i - 2
            rx = rxs[ind]
            ry = rys[ind]
            # print(sx, sy, rx, ry)
            # input('step')
            # MOVE_RIGHT
            if sx == rx - 1:
                self.rotate_agent(orientation, 'EAST')
                sx += 1
                orientation = 'EAST'
            # MOVE_LEFT
            elif sx == rx + 1:
                self.rotate_agent(orientation, 'WEST')
                orientation = 'WEST'
                sx -= 1
            # MOVE_NORTH
            elif sy == ry - 1:
                self.rotate_agent(orientation, 'SOUTH')
                orientation = 'SOUTH'
                sy += 1
            # MOVE_SOUTH
            elif sy == ry + 1:
                self.rotate_agent(orientation, 'NORTH')
                orientation = 'NORTH'
                sy -= 1
            else:
                print("error in path plan")
                return sx, sy
            self.plan.append('Forward')

            # self.env.step(self.env.actions_id['MOVE w'])
        # Rotate to face object at the end
        self.rotate_agent(orientation, ro)
        return sx, sy, self.plan

    def rotate_agent(self, start_o, goal_o):
        dir_vals = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}
        start_val = dir_vals[start_o]
        goal_val = dir_vals[goal_o]
        num_rots = goal_val - start_val
        if num_rots == 0:
            return
        elif num_rots == 1 or num_rots == -3:
            # self.plan.append('TURN 90')
            self.plan.append('Right')

            # self.env.step(self.env.actions_id['TURN 90'])
        elif num_rots == 2 or num_rots == -2:
            # self.plan.append('TURN 90')
            self.plan.append('Right')

            # self.plan.append('TURN 90')
            self.plan.append('Right')

            # self.env.step(self.env.actions_id['TURN 90'])
            # self.env.step(self.env.actions_id['TURN 90'])
        elif num_rots == 3 or num_rots == -1:
            # self.env.step(self.env.actions_id['TURN -90'])
            # self.plan.append('TURN -90')
            self.plan.append('Left')

    def get_action(self):
        # Returning none is indication that sequence is done
        if self.action_step >= len(self.plan):
            return None
        action = self.plan[self.action_step]
        self.action_step += 1
        return self.actions_id[action]

    def check_success(self, obs, info):
        # Need to give time for action to execute
        time.sleep(1)
        return self.check_success_func(obs, info)
        # return info['block_in_front'] == self.goal_type


# class ExtractRubberOperator:
#     def __init__(self, name, tappable_type, effect_set):
#         self.name = name
#         self.tappable_type = tappable_type
#         self.plan = []
#         self.action_step = 0
#         self.effect_set = effect_set
#         self.create_check_success_func = get_create_success_func_from_predicate_set(effect_set)
#         self.check_success_func = None

#     def reset(self, obs, info, env):
#         self.plan = []
#         self.actions_id = env.actions_id
#         self.action_step = 0
#         self.check_success_func = self.create_check_success_func(obs, info)

#         if get_world_quant(info, 'minecraft:log') == 0:
#             print('WARNING: No trees left in world, cannot extract rubber using this operator anymore - either need to tap something else or messed up earlier in trial')
#             return False
#         elif get_inv_quant(info, 'polycraft:tree_tap') == 0:
#             print('WARNING: Called ExtractRubber operator without posessing a tree tap')
#             #TODO: make agent go and pick up tap first
#         elif info['block_in_front']['name'] != self.tappable_type:
#             # TODO: make this able to goTo log?
#             print('WARN - Not in front of log, cannot extract rubber with current operator impl')
#             return False

#         posx = env.player['pos'][0]
#         posy = env.player['pos'][2]
#         orient = env.player['facing']
#         dir_vals = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}

#         #move to open space next to self and place tap where we originally stood
#         if env.binary_map[posy][posx-1] == 0:
#             self.rotate_agent(orient, 'WEST')
#             self.plan.append('MOVE w')
#             self.rotate_agent('WEST', 'EAST')
#             self.plan.append('PLACE_TREE_TAP')
#             self.plan.append('EXTRACT_RUBBER')
#         elif env.binary_map[posy][posx+1] == 0:
#             self.rotate_agent(orient, 'EAST')
#             self.plan.append('MOVE w')
#             self.rotate_agent('EAST', 'WEST')
#             self.plan.append('PLACE_TREE_TAP')
#             self.plan.append('EXTRACT_RUBBER')
#         elif env.binary_map[posy-1][posx] == 0:
#             self.rotate_agent(orient, 'NORTH')
#             self.plan.append('MOVE w')
#             self.rotate_agent('NORTH', 'SOUTH')
#             self.plan.append('PLACE_TREE_TAP')
#             self.plan.append('EXTRACT_RUBBER')
#         elif env.binary_map[posy+1][posx] == 0:
#             self.rotate_agent(orient, 'SOUTH')
#             self.plan.append('MOVE w')
#             self.rotate_agent('SOUTH', 'NORTH')
#             self.plan.append('PLACE_TREE_TAP')
#             self.plan.append('EXTRACT_RUBBER')
#         else:
#             print('Agent attempting to tap tree is surrounded by occupied spaces, did we block ourselves in?')
#             return False
#         return True

#     def rotate_agent(self, start_o, goal_o):
#         dir_vals = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}
#         start_val = dir_vals[start_o]
#         goal_val = dir_vals[goal_o]
#         num_rots = goal_val - start_val
#         if num_rots == 0:
#             return
#         elif num_rots == 1 or num_rots == -3:
#             self.plan.append('TURN 90')
#             # self.env.step(self.env.actions_id['TURN 90'])
#         elif num_rots == 2 or num_rots == -2:
#             self.plan.append('TURN 90')
#             self.plan.append('TURN 90')
#             # self.env.step(self.env.actions_id['TURN 90'])
#             # self.env.step(self.env.actions_id['TURN 90'])
#         elif num_rots == 3 or num_rots == -1:
#             # self.env.step(self.env.actions_id['TURN -90'])
#             self.plan.append('TURN -90')

#     def get_action(self):
#         # Returning none is indication that sequence is done
#         if self.action_step >= len(self.plan):
#             return None
#         action = self.plan[self.action_step]
#         self.action_step += 1
#         return self.actions_id[action]

#     def check_success(self, obs, info):
#         # Need to give time for action to execute
#         time.sleep(1)
#         return self.check_success_func(obs, info)
#         # return info['block_in_front'] == self.goal_type
