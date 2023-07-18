#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:10:05 2021

@author: edmir
@author: vincent
"""
import csv

import numpy as np
import shapely.geometry as sg
import math
from descartes.patch import PolygonPatch
from figures import BLUE, GREEN, SIZE, set_limits, plot_coords, color_isvalid
import matplotlib.pyplot as plt
from shapely.wkt import loads
from poly import Poly
from vru import VRU
from shapely.ops import unary_union

class Vehicle():
	wedge_nr_pnts = 8
	wedge_max_angle = 60
	wedge_min_len = 20
	shadow_nr_pnts = 100
	radar_len = 55
	
	def __init__(self, x, y, heading, vel_x, vel_y, acc_x, acc_y, track_frame, length=4, width=2, trackId=0, scaling_factor=1):
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
		
		self.trackId = trackId

		self.other_vru = []
		self.other_veh = []

		self.vru_list = []

		self.risk_counter = 0
		self.in_range_counter = 0

		self.scaling_factor = scaling_factor

		self.ax1 = None
		self.fig = None

		self.last_track_frame = track_frame

		self.static_polygons = []
		self.update_postion(x, y, heading, vel_x, vel_y, acc_x, acc_y, track_frame)
		
		self.vru_tracking_log = {}

		self.xVarDist, self.yVarDist = load_covariance_map()

	def write_log(self):
		print(self.vru_tracking_log)
		if self.vru_tracking_log != {}:
			write_log_to_csv(self.vru_tracking_log, 'log_file' + str(self.trackId) + '.csv')

	def update_postion(self, x, y, heading, vel_x, vel_y, acc_x, acc_y, track_frame):
		self.pos = [x, y]
		self.x = x
		self.y = y
		self.heading = heading
		self.update_vehicle()
		self.lines = self.update_lines()
		self.acc = self.calculate_acc(acc_x, acc_y)
		self.velocity = self.calculate_velocity(vel_x, vel_y)
		self.last_track_frame = track_frame
		self.safetyCriticalAreaRadius = self.calculate_critical_radius()
		self.update_shadow()

	def calculate_velocity(self, vel_x, vel_y):
		current_velocity = np.sqrt(vel_x ** 2 + vel_y ** 2) * 3.6
		return abs(float(current_velocity))
	
	def calculate_acc(self, acc_x, acc_y):
		return np.sqrt(acc_x ** 2 + acc_y ** 2)

	def update_lines(self):
		if len(self.corner_xs) > 0:
			line = []
			for i in np.arange(1, len(self.corner_xs)):
				line.append([[self.corner_xs[i], self.corner_ys[i]], [self.corner_xs[i-1], self.corner_ys[i-1]]])
			self.lines = line
			return self.lines
		else:
			print("Error")

	def update_vehicle(self):
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
		#dst = 1
		dspl = math.degrees(math.atan((self.width/2) / (self.length/2)))
		heading = self.heading
		degrs = [heading+dspl, heading-dspl, heading+dspl+180, heading-dspl+180]
		return self.get_pnt(init_v, dst, degrs)

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




		# 		# Check intersection with polygons
			for polygon in self.static_polygons:

				min_x = min(polygon.corner_xs)
				max_x = max(polygon.corner_xs)
				min_y = min(polygon.corner_ys)
				max_y = max(polygon.corner_ys)

				angle1_poly = math.degrees(math.atan2(max_y - self.y, min_x - self.x))
				angle2_poly = math.degrees(math.atan2(min_y - self.y, max_x - self.x))
				min_angle = min(angle1_poly, angle2_poly)
				max_angle = max(angle1_poly, angle2_poly)

				if (angle > min_angle and angle < max_angle):
					#if other_veh != self and self.in_range(other_veh):
					for ln in (polygon.get_lines()):
						intersect = line_intersection(line_radar, ln)
						if intersect == None:
							new_distance = 1000000
						else:
							intersect = (round(intersect[0], 4), round(intersect[1], 4))
							new_distance = calc_distance([self.x, self.y], intersect)
						if new_distance < distance:
							distance = new_distance
							last_point = intersect


			points.append(last_point)

		self.shadow_circle = points	

	def update_vru(self, x, y, heading, track_frame, type, id):
		found_vru = [item for item in self.other_vru if item.trackId == id]
		#print(found_vru)
		if len(found_vru) > 0:
			found_vru = found_vru[0]
			#print("old position: " + str(found_vru.x) + "," + str(found_vru.y) + "\n")
			found_vru.update_position(x, y, heading, track_frame)
			#print("new position: " + str(found_vru.x) + "," + str(found_vru.y) + "\n")
		else:
			found_vru = VRU(x, y, heading, track_frame, type, id)
			print("vehicle id: " + str(self.trackId))
			print("VRU added at " + str(x) + "," + str(y) + " width id: " + str(id) +  "\n")
			self.other_vru.append(found_vru)


	def set_static_polygons(self, polygons):
		for poly in polygons:
			#poly = sg.Polygon(poly)
			poly = Poly(poly)
			self.static_polygons.append(poly)

	def get_lines(self):
		return self.lines

	def in_los(self, obj):
		poly = obj.get_outer_poly()
		if self.shadow_circle != []:
			circle_poly = sg.Polygon(self.shadow_circle)
			if circle_poly.covers(poly):
			#if circle_poly.intersects(poly):
				return True
			else:
				return False

	def calculate_critical_radius(self):
		t = 2.5
		safetyCriticalAreaRadius = (t * abs(self.velocity) + 0.5 * abs(self.acc * t ** 2)) #+ ((self.velocity/3.6 + self.acc)** 2 / 2 * 2)
		#safetyCriticalAreaRadius = (self.velocity) + ((self.velocity) ** 2 / 4)
		return safetyCriticalAreaRadius


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
		# Create a polygon for the vehicle based on the corner coordinates
		vehicle_corners = list(zip(self.corner_xs, self.corner_ys))
		vehicle_poly = sg.Polygon(vehicle_corners)

		# Adding a buffer to the vehicle polygon
		buffer_size = 5  # Define your buffer size here
		vehicle_poly = vehicle_poly.buffer(buffer_size)

		self.wedge = points
		self.wedge_poly = sg.Polygon(self.wedge)

		# Combine the wedge polygon with the buffered vehicle polygon
		self.combined_poly = unary_union([self.wedge_poly, vehicle_poly])




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
		#self.other_vru = []


	def visualize(self, min_x, max_x, min_y, max_y):
		self.update_shadow()
		if self.fig == None:
			self.fig = plt.figure()
		if self.ax1 == None:
			self.ax1 = self.fig.add_subplot(1,1,1)
		else:
			self.ax1.clear()
		self.ax1.plot(self.corner_xs, self.corner_ys, color="black")


		shadow_poly = sg.Polygon(self.shadow_circle)
		patch_shadow = PolygonPatch(shadow_poly, facecolor="green", edgecolor="green", alpha=0.1)
		self.ax1.add_patch(patch_shadow)
		for veh in self.other_veh:
			self.ax1.plot(veh.corner_xs, veh.corner_ys, color="grey")
		for vru in self.other_vru:
			print("VRU Track: " + str(vru.trackId))
			print(vru.x)
			print(vru.y)
			print("Speed")
			print(vru.speedX)
			print(vru.speedY)
			print(vru.heading)
			patch_vru = PolygonPatch(vru.get_outer_poly(), facecolor="red", edgecolor="red", alpha=1)
			self.ax1.add_patch(patch_vru)
		buffer = 80
		self.ax1.set_xlim([min_x - buffer, max_x + buffer])
		self.ax1.set_ylim([min_y - buffer, max_y + buffer])
		plt.pause(0.01)

	def update_all_vrus(self):
		for vru in self.other_vru:
			if vru.last_track_frame != self.last_track_frame:
				self.remove_vru(vru)
				#vru.increase_miss_tracking_counter()
				#vru.currently_tracked = False


	# def noisy_measurement(self):
	# 	for vru in self.other_vru :
	# 		if self.in_range(vru):
	# 			if self.in_los(vru):
	# 				print("update tracking")
	# 				distance = calc_distance([self.x, self.y], vru.get_position())
	# 				noisy_position = add_sensor_noise(vru.x, vru.y, 2, 2)
	# 				vru_tracking_position = vru.update_tracking(noisy_position[0], noisy_position[1])
	# 			else:
	# 				#distance = calc_distance([self.x, self.y], [vru.get_position()])
	# 				print("not in LOS.")# distance: " + str(distance))
	# 				vru_tracking_position = vru.predict_tracking()
	# 				vru.increase_miss_tracking_counter()
	# 		else:
	# 			self.remove_vru(vru)

	def remove_vru(self, vru):
		self.other_vru.remove(vru)

	def process_risk(self):
		self.safetyCriticalAreaRadius = self.calculate_critical_radius()

		#for vru in self.other_vru:


	def noisy_measurement(self):
		ignored = False
		for vru in self.other_vru:
			# Update the tracking accuracy log
			if vru.trackId not in self.vru_tracking_log:
				self.vru_tracking_log[vru.trackId] = {'positions': [], 'errors': [], 'measurementErrors' : [], 'misses': 0, 'tracked': []}
			if self.in_range(vru):
				if self.in_los(vru):
					distance = calc_distance([self.x, self.y], vru.get_position())
					x_var = self.xVarDist[distance - (distance % 10)] if distance < 70 else list(self.xVarDist.values())[-1]
					y_var = self.yVarDist[distance - (distance % 10)] if distance < 70 else list(self.yVarDist.values())[-1]

					noisy_position = add_sensor_noise(vru.x, vru.y, x_var , y_var)
					vru.update_tracking(noisy_position[0], noisy_position[1])
				else:
				#	if vru.miss_tracking_counter > 4:
				#		print("removing VRU with id " + str(vru.trackId) + "\n")
				#		print(self.vru_tracking_log[vru.trackId])
				#		self.remove_vru(vru)
				#		deleted = True
				#	else:
					print("not in LOS.")
					vru.predict_tracking()
					vru.increase_miss_tracking_counter()
					# Log a tracking miss
					self.vru_tracking_log[vru.trackId]['misses'] += 1
			else:
				print("ignoring VRU with id " + str(vru.trackId) + "\n")
				#self.remove_vru(vru)
				ignored = True

			if not ignored:
				if vru.lastMeasurementX != None:
					ground_truth = [vru.x, vru.y]
					vru_tracking_position = [vru.trackingX, vru.trackingY]
					error = calc_distance(ground_truth, vru_tracking_position)

					measurement_error = calc_distance(ground_truth, [vru.lastMeasurementX, vru.lastMeasurementY])
					self.vru_tracking_log[vru.trackId]['positions'].append(vru_tracking_position)
					self.vru_tracking_log[vru.trackId]['errors'].append(error)
					self.vru_tracking_log[vru.trackId]['measurementErrors'].append(measurement_error)
					self.vru_tracking_log[vru.trackId]['tracked'].append(vru.currently_tracked)

def load_covariance_map():
	dist = "0,10,20,30,40,50,60,70"
	x = "0.1,0.15,0.18,0.22,0.3,0.45,0.55,0.65"
	#y = "0.0904478,0.00886361,0.00923802,0.0218104,0.0106392,0.0110924,0.0172919,0.0245167"
	y = x
	distances = dist.split(",")
	x_vars = x.split(",")
	y_vars = y.split(",")

	x_variance = {}
	y_variance = {}

	for i in range(len(distances)):
		x_variance[float(distances[i])] = float(x_vars[i])
		y_variance[float(distances[i])] = float(y_vars[i])

	return x_variance, y_variance

def line_intersection(ln1, ln2):

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

def add_sensor_noise(x_true, y_true, sigma_x, sigma_y):
    """
    Function to add Gaussian noise to a 2D ground truth measurement.

    Args:
    x_true: Ground truth x-coordinate
    y_true: Ground truth y-coordinate
    sigma_x: Standard deviation of the noise in the x direction
    sigma_y: Standard deviation of the noise in the y direction

    Returns:
    x_noisy: Noisy x-coordinate
    y_noisy: Noisy y-coordinate
    """

    # Add Gaussian noise
    x_noisy = x_true + np.random.normal(0, sigma_x)
    y_noisy = y_true + np.random.normal(0, sigma_y)

    return x_noisy, y_noisy


def write_log_to_csv(log_data, filename):
    fieldnames = ['Track ID', 'Error', 'MeasurementError', 'Tracked']

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)

        for track_id, data in log_data.items():
            #positions = data['positions']
            errors = data['errors']
            measurement_errors = data['measurementErrors']
            tracked = data['tracked']

            # Ensure all lists have the same length
            num_entries = min(len(errors), len(measurement_errors), len(tracked))

            for i in range(num_entries):
                row = [track_id, errors[i], measurement_errors[i], tracked[i]]
                writer.writerow(row)
