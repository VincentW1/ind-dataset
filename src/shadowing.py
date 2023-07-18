#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:58:14 2021

@author: edmir
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from vehicle import Vehicle as vehicle
import matplotlib.animation as animation
import shapely.geometry as sg
from descartes.patch import PolygonPatch
from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid

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

def get_pnt(init_point, length, degrees):
	x = init_point[0]
	y = init_point[1]	
	points =	[]
#	degrees = degrees[:2]
	for angle in degrees:
		endy = y + length * math.sin(math.radians(angle))
		endx = x + length * math.cos(math.radians(angle))
		points.append([endx, endy])
		
	points.append(points[0])	
	return points 


def crcl_draw(init_pnt, length, nr_pnts=360):
	points = []
	#x = init_pnt[0]
	#y = init_pnt[1]
	x,y = init_pnt
#	points.append(init_pnt)
	for angle in np.arange(0, 361, 360/nr_pnts):
		endy = y + length * math.sin(math.radians(angle))
		endx = x + length * math.cos(math.radians(angle))
		points.append([endx, endy])
	return points

def wedge_draw(init_pnt, heading, length, max_angle, nr_pnts=20):
	points = []
	x,y = init_pnt
	points.append([x, y])
	step_size = nr_pnts/max_angle
	start = heading - max_angle/2
	stop = heading + max_angle/2
	for angle in np.arange(start, stop+step_size, step_size):
		endy = y + length * math.sin(math.radians(angle))
		endx = x + length * math.cos(math.radians(angle))
		points.append([endx, endy])
	return points

def veh_draw(init_v, heading):
	len_veh = 4
	width_veh = 2
	dst = math.sqrt((len_veh/2)**2 + (width_veh/2)**2)
	dspl = math.degrees(math.atan((width_veh/2) / (len_veh/2)))
	heading = heading
	degrs = [heading+dspl, heading-dspl, heading+dspl+180, heading-dspl+180]
	return get_pnt(init_v, dst, degrs)
# exp
def veh_transform(v, heading):
	corner = {}
	corner[0] = sg.Point(v.x + v.width/2, v.y + v.length/2) - sg.Point(v.x, v.y)
	corner[1] = sg.Point(v.x + v.width/2, v.y - v.length/2) - sg.Point(v.x, v.y)
	corner[2] = sg.Point(v.y - v.width/2, v.y + v.length/2) - sg.Point(v.x, v.y)
	corner[3] = sg.Point(v.y - v.width/2, v.y - v.length/2) - sg.Point(v.x, v.y)

	result = []
	for c in corner.values():
		temp = c - sg.Point(v.x, v.y)
		rotatedX = temp.x*math.cos(heading) - temp.y*math.sin(heading)
		rotatedY = temp.x*math.sin(heading) + temp.y*math.cos(heading)
		c.x += v.x
		c.y += v.y
		#c = c + sg.Point(v.x, v.y)
		result.append(c)
	return result

def in_range(veh1, veh2):
	pos1 = veh1.pos
	pos2 = veh2.pos
	dst = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
	check = veh1.radar_len + np.sqrt(veh1.length**2 + veh1.width**2)
	if dst < check:
		return True
	else:
		return False
	
def calc_distance(pos1, pos2):
	dst = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
	return dst
	
def shadow_draw(ego_vehicle, vehis_obj, nr_pnts=360):
	# vehi is ego-vehicle
	# vehis is other vehicle
	points = []
	x = ego_vehicle.pos[0]
	y = ego_vehicle.pos[1]
	length = ego_vehicle.radar_len
	
	for angle in np.arange(0, 361, 360/nr_pnts):
		endy = y + length * math.sin(math.radians(angle))
		endx = x + length * math.cos(math.radians(angle))
		line_radar = [[x,y], [endx, endy]]
		distance = ego_vehicle.radar_len
		last_point = [endx, endy]
		for other_veh in vehis_obj:
			if other_veh != ego_vehicle and in_range(ego_vehicle, other_veh):
				for ln in (other_veh.get_lines()):
					intersect = line_intersection(line_radar, ln)
#					intersect = (round(intersect[0], 4),round(intersect[1], 4))
					if intersect == None:
						new_distance = 1000000
					else:
						#print("Intersection with Vehicle at (" + str(other_veh.x) + "," + str(other_veh.y) + ")")
						intersect = (round(intersect[0], 4),round(intersect[1], 4))
						new_distance = calc_distance([x,y], intersect)
					if new_distance < distance:
						distance = new_distance
						last_point = intersect
		points.append(last_point)	
		
	return points	

# def in_los(ego_vehicle, poly):
# 		# vehi is ego-vehicle
# 	# vehis is other vehicle
# 	points = []
# 	x = ego_vehicle.pos[0]
# 	y = ego_vehicle.pos[1]
# 	length = ego_vehicle.radar_len
	
# 	for angle in np.arange(0, 361, 360/nr_pnts):
# 		endy = y + length * math.sin(math.radians(angle))
# 		endx = x + length * math.cos(math.radians(angle))
# 		line_radar = [[x,y], [endx, endy]]
# 		distance = ego_vehicle.radar_len
# 		last_point = [endx, endy]
# 		#for other_veh in vehis_obj:
# 		#	if other_veh != ego_vehicle and in_range(ego_vehicle, other_veh):


# 		for ln in poly:
# 			intersect = line_intersection(line_radar, ln)
# 			if intersect == None:
# 				new_distance = 1000000
# 			else:
# 				intersect = (round(intersect[0], 4),round(intersect[1], 4))
# 				new_distance = calc_distance([x,y], intersect)
# 				return True
# 			if new_distance < distance:
# 				distance = new_distance
# 				last_point = intersect

# 	return False

vehs = []
#	def __init__(self, pos, heading, radar_len, length=4, width=2):
vehs.append(vehicle([0,0], 0, 20))
vehs.append(vehicle([5, 0], 0, 20))
vehs.append(vehicle([10,0], 40, 20))

colors = ["red", "green", "yellow"]
fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)

def animate(i):
	for veh in [vehs[2]]:
		#veh.pos = [veh.pos[0]+i*0.1, veh.pos[1]+i*0.1]
		veh.pos = [veh.pos[0] + 0.1 * math.cos(math.radians(veh.heading)), veh.pos[1] + 0.1 * math.sin(math.radians(veh.heading))]
		
	for veh in vehs:
		corners = veh_draw(veh.pos, veh.heading)
		veh.corner_xs = [circ[0] for circ in corners]
		veh.corner_ys = [circ[1] for circ in corners]
		veh.update_lines()
	

	for veh in vehs:
		circle = crcl_draw(veh.pos, veh.radar_len)
		veh.wedge = wedge_draw(veh.pos, veh.heading, veh.radar_len, 60)
		veh.radar_xs = [circ[0] for circ in circle]
		veh.radar_ys = [circ[1] for circ in circle]

	for veh in vehs:
		circle = shadow_draw(veh, vehs)
		veh.shadow_xs = [circ[0] for circ in circle]
		veh.shadow_ys = [circ[1] for circ in circle]
		veh.shadow_circle = circle
#	lim = 40
	ax1.clear()
	poly = sg.box(0.5, 0.5, 3, 3, ccw=True)
	x, y = poly.exterior.xy
	ax1.plot(x, y, linewidth=2, color="black")
	c = 0
	print("---")
	for veh in vehs:
		ax1.plot(veh.corner_xs, veh.corner_ys, color= colors[c])
		ax1.plot(veh.radar_xs, veh.radar_ys, linewidth=0.2, color= colors[c])
		wedge_poly = sg.Polygon(veh.wedge)
		plot_coords(ax1, wedge_poly.exterior)
		patch = PolygonPatch(wedge_poly, facecolor=color_isvalid(wedge_poly), edgecolor=color_isvalid(wedge_poly, valid=BLUE), alpha=0.5, zorder=2)
		ax1.add_patch(patch)
		#ax1.plot(veh.shadow_xs, veh.shadow_ys,"-.", linewidth = 2.5, color= colors[c])
		#print(veh.in_los(poly))

		c += 1

	

	 
	
		
ani = animation.FuncAnimation(fig, animate, interval=1000) 
plt.show()

