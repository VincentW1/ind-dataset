#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:10:05 2021

@author: edmir
@author: vincent
"""
import numpy as np
import shapely.geometry as sg
import math
from descartes.patch import PolygonPatch
from figures import BLUE, GREEN, SIZE, set_limits, plot_coords, color_isvalid
import matplotlib.pyplot as plt

class Vehicle():
	wedge_nr_pnts = 8
	wedge_max_angle = 60
	wedge_min_len = 20
	shadow_nr_pnts = 500
	radar_len = 50
	
	def __init__(self, x, y, heading, vel_x, vel_y, acc_x, acc_y, length=4, width=2, trackId=0, scaling_factor=1):
		self.trackId = trackId
		self.pos = [x, y]
		self.x = x
		self.y = y

		self.heading = heading
		self.length = length
		self.width = width
		#self.wedge = self.update_wedge()
		
		self.corner_xs = []
		self.corner_ys = []
		
		self.radar_xs = []
		self.radar_ys = []
		
		self.shadow_xs = []
		self.shadow_ys = []
		
		#self.update_vehicle()

		#self.lines = self.update_lines()
		
		self.trackId = trackId

		self.other_vru = []
		self.other_veh = []

		self.vru_list = []

		#self.update_shadow()

		self.risk_counter = 0
		self.in_range_counter = 0

		self.scaling_factor = scaling_factor

		#self.acc = self.calculate_acc(acc_x, acc_y)
		#self.velocity = self.calculate_velocity(vel_x, vel_y)
		#self.safetyCriticalAreaRadius = self.calculate_critical_radius()
		#self.update_wedge()
		self.ax1 = None
		self.fig = None
		self.update_postion(x, y, heading, vel_x, vel_y, acc_x, acc_y)
	
	def update_postion(self, x, y, heading, vel_x, vel_y, acc_x, acc_y):
		self.pos = [x, y]
		self.x = x
		self.y = y
		self.heading = heading
		self.update_vehicle()
		self.lines = self.update_lines()
		#self.update_shadow()
		self.acc = self.calculate_acc(acc_x, acc_y)
		self.velocity = self.calculate_velocity(vel_x, vel_y)
		self.safetyCriticalAreaRadius = self.calculate_critical_radius()
		#self.update_wedge()

	def calculate_velocity(self, vel_x, vel_y):
		current_velocity = np.sqrt(vel_x ** 2 + vel_y ** 2) * 3.6
		return abs(float(current_velocity))
	
	def calculate_acc(self, acc_x, acc_y):
		return np.sqrt(acc_x ** 2 + acc_y ** 2)

	def calculate_critical_radius(self):
		t = 2.5
		safetyCriticalAreaRadius = (t * abs(self.velocity) + 0.5 * abs(self.acc * t ** 2)) #+ ((self.velocity/3.6 + self.acc)** 2 / 2 * 2)
		#safetyCriticalAreaRadius = (self.velocity) + ((self.velocity) ** 2 / 4)
		return safetyCriticalAreaRadius

	def update_lines(self):
		if len(self.corner_xs)> 0:
			line = []
			for i in np.arange(1, len(self.corner_xs)):
				line.append([[self.corner_xs[i], self.corner_ys[i]], [self.corner_xs[i-1], self.corner_ys[i-1]]])
			self.lines = line
			return self.lines
		else:
			print("Error")

	def update_vehicle(self):
		# TODO
		corners = self.veh_draw(self.pos, self.heading)
		self.corners = corners
		self.corner_xs = [circ[0] for circ in corners]
		self.corner_ys = [circ[1] for circ in corners]

	def get_pnt(this, init_point, length, degrees):
		x = init_point[0]
		y = init_point[1]	
		points =	[]
		for angle in degrees:
			endy = y + length * math.sin(math.radians(angle))
			endx = x + length * math.cos(math.radians(angle))
			points.append([endx, endy])
			
		points.append(points[0])	
		return points 	

	def veh_draw(self, init_v, heading):
		dst = math.sqrt((self.length/2)**2 + (self.width/2)**2)
		dst = 1
		dspl = math.degrees(math.atan((self.width/2) / (self.length/2)))
		heading = self.heading
		degrs = [heading+dspl, heading-dspl, heading+dspl+180, heading-dspl+180]
		return self.get_pnt(init_v, dst, degrs)

	def update_wedge(self):
		wedge_length = max(self.safetyCriticalAreaRadius, self.wedge_min_len)
		points = []
		x,y = self.x, self.y
		points.append([x, y])
		step_size 	= self.wedge_nr_pnts/self.wedge_max_angle
		start 		= self.heading - self.wedge_max_angle/2
		stop 		= self.heading + self.wedge_max_angle/2
		for angle in np.arange(start, stop+step_size, step_size):
			endy = y + wedge_length * math.sin(math.radians(angle))
			endx = x + wedge_length * math.cos(math.radians(angle))
			points.append([endx, endy])
		self.wedge = points
		self.wedge_poly = sg.Polygon(self.wedge)

	def update_shadow(self):
		points = []
		length = self.radar_len
		
		for angle in np.arange(-180, 180, 360/self.shadow_nr_pnts):
			endy = self.y + length * math.sin(math.radians(angle))
			endx = self.x + length * math.cos(math.radians(angle))
			line_radar = [[self.x,self.y], [endx, endy]]
			distance = length
			last_point = [endx, endy]
			for other_veh in self.other_veh:

				min_x = min(other_veh.corner_xs)
				max_x = max(other_veh.corner_xs)
				min_y = min(other_veh.corner_ys)
				max_y = max(other_veh.corner_ys) 

				angle1_veh = math.degrees(math.atan2(max_y - self.y, min_x - self.x))
				angle2_veh = math.degrees(math.atan2(min_y - self.y, max_x - self.x))
				min_angle = min(angle1_veh, angle2_veh)
				max_angle = max(angle1_veh, angle2_veh)

				if(angle > min_angle and angle < max_angle):
					if other_veh != self and self.in_range(other_veh):
						for ln in (other_veh.get_lines()):
							intersect = line_intersection(line_radar, ln)
							if intersect == None:
								new_distance = 1000000
							else:
								intersect = (round(intersect[0], 4),round(intersect[1], 4))
								new_distance = calc_distance([self.x,self.y], intersect)
							if new_distance < distance:
								distance = new_distance
								last_point = intersect
			points.append(last_point)	
			
		self.shadow_circle = points	

	def get_lines(self):
		return (self.lines)

	def in_los(self, obj):
		poly = obj.get_outer_poly()
		if self.shadow_circle != []:
			circle_poly = sg.Polygon(self.shadow_circle)
			if circle_poly.intersects(poly):
				return True
			else:
				return False

	def in_critical_range(self, obj):
		poly = obj.get_outer_poly()
		if self.wedge != []:
			if self.wedge_poly.intersects(poly):
				return True
			else:
				return False

	def get_outer_poly(self):
		poly = sg.Polygon(self.corners)
		return poly

	def in_range(self, veh2):
		pos1 = self.pos
		pos2 = veh2.pos
		dst = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
		check = self.radar_len + np.sqrt(self.length**2 + self.width**2)
		if dst < check:
			return True
		else:
			return False

	def add_other_veh(self, veh):
		self.other_veh.append(veh)

	def add_other_vru(self, vru):
		self.other_vru.append(vru)
	
	def clear_other_objects(self):
		self.other_veh = []
		self.other_vru = []

	def calculate_risk(self):
		self.update_shadow()
		self.update_wedge()
		for vru in self.other_vru:
			if self.in_critical_range(vru):
				if not self.in_los(vru):
					self.risk_counter += 1
					vru.risk_counter += 1
				self.in_range_counter += 1
				vru.in_critical_range += 1
				print("VRU #" + str(vru.trackId) + " was in critical range " + str(vru.in_critical_range) + " times")
	def visualize(self, min_x, max_x, min_y, max_y):
		self.update_shadow()
		self.update_wedge()
		if self.fig == None:
			self.fig = plt.figure()
		if self.ax1 == None:
			self.ax1 = self.fig.add_subplot(1,1,1)
		else:
			self.ax1.clear()
		self.ax1.plot(self.corner_xs, self.corner_ys, color="black")
		#plot_coords(self.ax1, self.wedge_poly.exterior)
		patch_wedge = PolygonPatch(self.wedge_poly, facecolor="blue", edgecolor="blue", alpha=0.5)
		self.ax1.add_patch(patch_wedge)
		#self.ax2 = fig.add_subplot(1,1,1)
		shadow_poly = sg.Polygon(self.shadow_circle)
		#plot_coords(self.ax2, shadow_poly.exterior)
		patch_shadow = PolygonPatch(shadow_poly, facecolor="green", edgecolor="green", alpha=0.1)
		self.ax1.add_patch(patch_shadow)
		for veh in self.other_veh:
			self.ax1.plot(veh.corner_xs, veh.corner_ys, color="grey")
		for vru in self.other_vru:
			#self.ax1.plot(veh.corner_xs, veh.corner_ys, color="grey")
			patch_vru = PolygonPatch(vru.get_outer_poly(), facecolor="red", edgecolor="red", alpha=1)
			self.ax1.add_patch(patch_vru)
		buffer = 80
		self.ax1.set_xlim([min_x - buffer, max_x + buffer])
		self.ax1.set_ylim([min_y - buffer, max_y + buffer])
		plt.pause(0.01)
def line_intersection(ln1, ln2):
		""" returns a (x, y) tuple or None if there is no intersection """
#		(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2)
		Ax1 = ln1[0][0]
		Ay1 = ln1[0][1]
		Ax2 = ln1[1][0] 
		Ay2 = ln1[1][1]
		Bx1 = ln2[0][0]
		By1 = ln2[0][1]
		Bx2 = ln2[1][0]
		By2 = ln2[1][1]
		
		d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
		if d:
				uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
				uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
		else:
				return
		if not(0 <= uA <= 1 and 0 <= uB <= 1):
				return
		x = Ax1 + uA * (Ax2 - Ax1)
		y = Ay1 + uA * (Ay2 - Ay1)
 
		return x, y

def calc_distance(pos1, pos2):
	dst = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
	return dst

