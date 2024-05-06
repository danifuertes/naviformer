"""

D* grid planning

author: Nirnay Roy

See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""
import math
import numpy as np

from sys import maxsize

import matplotlib.pyplot as plt

show_animation = False


class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state
        self.h = 0
        self.k = 0

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))

    def set_state(self, state):
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_obstacle(self, point_list):
        for x, y in point_list:
            x, y = round(x), round(y)
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue

            self.map[x][y].set_state("#")


class DStar:
    def __init__(self, obs, margin=5, scale=100):
        self.create_obs_map(obs, margin=margin, scale=scale)
        self.set_obs_map()

    def create_obs_map(self, obs, margin=5, scale=100, *args, **kwargs):
        self.ox, self.oy, self.scale = [], [], scale

        # Define obstacles
        for ob in obs:
            cx, cy, r = ob
            x, y = np.meshgrid(np.linspace(0, scale, scale + 1), np.linspace(0, scale, scale + 1))
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            points_in_circle = np.where(np.logical_and(2 * r / 3 <= distance, distance <= r))
            x_coords = x[points_in_circle]
            self.ox = [*self.ox, *x_coords]
            y_coords = y[points_in_circle]
            self.oy = [*self.oy, *y_coords]

        # Define margins as obstacles
        for ob in range(-margin, scale + margin):
            self.ox.append(ob)
            self.oy.append(-margin)

            self.ox.append(-margin)
            self.oy.append(ob)

            self.ox.append(ob)
            self.oy.append(scale + margin)

            self.ox.append(scale + margin)
            self.oy.append(ob)
    
    def set_obs_map(self):
        self.open_list = set()
        self.resolution = 2
        self.map = Map(self.scale + 1, self.scale + 1)
        self.map.set_obstacle([(i, j) for i, j in zip(self.ox, self.oy)])
    
    def process_state(self):
        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()
        self.remove(x)

        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(y, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    # import numpy as np
    # ab = np.zeros((len(self.map.map), len(self.map.map[0])))
    # for a in range(len(self.map.map)):
    #     for b in range(len(self.map.map[a])):
    #         if self.map.map[a][b].state == '#':
    #             ab[b, a] = 1
    # plt.scatter(sx, sy)
    # plt.scatter(ex, ey)
    # plt.imshow(ab, origin='lower')
    # plt.show()
    def planning(self, sx, sy, ex, ey, limit, **kwargs):
        sx, sy = round(sx), round(sy)
        ex, ey = round(ex), round(ey)
        start = self.map.map[sx][sy]
        end = self.map.map[ex][ey]
        if sx == ex and sy == ey:
            return [ex], [ey], True

        rx = []
        ry = []

        self.insert(end, 0.0)

        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")
        s = start
        s = s.parent
        s.set_state("e")
        tmp = start

        count, success = 0, True
        while tmp != end:
            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)
            if show_animation:
                plt.plot(rx, ry, "-r")
                plt.pause(0.01)
            if tmp.parent.state == "#":
                self.modify(tmp)
                continue
            tmp = tmp.parent
            count += 1
            if count >= limit:
                success = False
                break
        tmp.set_state("e")

        return rx, ry, success

    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break


def main():
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)
    print([(i, j) for i, j in zip(ox, oy)])

    start = [10, 10]
    goal = [50, 50]
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.axis("equal")

    d_star = DStar(ox, oy)
    rx, ry = d_star.planning(*start, *goal)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()
