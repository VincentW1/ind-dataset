import math

import numpy as np


class Poly():


    def __init__(self, coordinates):

        self.corner_xs = []
        self.corner_ys = []
        self.corners = coordinates
        self.corner_xs = [c[0] for c in coordinates]
        self.corner_ys = [c[1] for c in coordinates]


        self.lines = self.update_lines()


    def update_lines(self):
        if len(self.corner_xs) > 0:
            line = []
            for i in np.arange(1, len(self.corner_xs)):
                line.append([[self.corner_xs[i], self.corner_ys[i]], [self.corner_xs[i - 1], self.corner_ys[i - 1]]])
            self.lines = line
            return self.lines

    def get_lines(self):
        return self.lines
    # def update_poly(self):
    #     corners = self.poly_draw(self.pos)
    #     self.corners = corners
    #     self.corner_xs = [circ[0] for circ in corners]
    #     self.corner_ys = [circ[1] for circ in corners]
    #
    # def poly_draw(self, init_v, heading=0):
    #     dst = math.sqrt((self.length / 2) ** 2 + (self.width / 2) ** 2)
    #     # dst = 1
    #     dspl = math.degrees(math.atan((self.width / 2) / (self.length / 2)))
    #     heading = self.heading
    #     degrs = [heading + dspl, heading - dspl, heading + dspl + 180, heading - dspl + 180]
    #     return self.get_pnt(init_v, dst, degrs)
    #
    # def get_pnt(this, init_point, length, degrees):
    #     x = init_point[0]
    #     y = init_point[1]
    #     points = []
    #     for angle in degrees:
    #         endy = y + length * math.sin(math.radians(angle))
    #         endx = x + length * math.cos(math.radians(angle))
    #         points.append([endx, endy])
    #
    #     points.append(points[0])
    #     return points