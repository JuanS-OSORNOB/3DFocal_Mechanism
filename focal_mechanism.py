#!/usr/bin/env python3
# Last revised: 19/06/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import argparse
import numpy as np
import pandas as pd
import os, sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import radians, sin, cos, isclose, asin, atan2
from vector_math import vectors, fm_quadrant, fm_points, shorten_line
from plotcoords import circle_arc
from datautils import parse_file

#command line arguments
parser = argparse.ArgumentParser(description='Plot 3D focal mechanisms')
parser. add_argument('filename', nargs = '?')
parser.add_argument('-r')

args = parser.parse_args()




def plot_circle(radius, center, vecs, ax, scale_factors, fault_color = 'black', auxiliary_color = 'blue',
				degrees = True):
	strike = vecs['strike']
	dip = vecs['dip']
	normal = vecs['normal']
	null = vecs['B']

	#fault plane, defined by strike and dip vectors which are orthogonal and both in the plane
	x, y, z = circle_arc(strike, dip, 0, 2 * np.pi, center, scale_factors, radius)
	ax.plot(x, y, z, color = fault_color, linewidth = 2)
	
	#auxiliary plane, defined by normal and null vectors which are orthogonal and both in the plane
	x, y, z = circle_arc(normal, null, 0, 2 * np.pi, center, scale_factors, radius)
	ax.plot(x, y, z, color = auxiliary_color, linewidth = 2)

def plot_vector(radius, center, vec, ax, scale_factors, color):
	v = vec * scale_factors
	ax.quiver(*center, *v, colors = color, length = radius)

def vec_to_angles(vector):
	'''takes an xyz vector and returns bearing (degrees clockwise from y axis) and
	plunge (degrees below horizontal plane) angles.'''
	
	x, y, z = vector
	mag = np.linalg.norm(vector)
	bearing = atan2(x, y) * 180/np.pi
	plunge = -asin(z/mag) * 180/np.pi

	if bearing<0:
		bearing=360+bearing
	return bearing, plunge

def print_vectors(vecs):
	'''Takes a dict of xyz vectors, prints the vector type, xyz vector, and plunge/bearing format.'''

	textstring = '{0}: <{1},{2},{3}>, bearing: {4}°, plunge: {5}°'
	for v in vecs:
		bearing, plunge = vec_to_angles(vecs[v])
		#shorten to two decimal places
		shortened = ['{:.2f}'.format(x) for x in [*vecs[v], bearing, plunge]]
		vecs_FM=textstring.format(v, *shortened)
		print(vecs_FM)
	return bearing, plunge

def scale_beachballs(beachball_list, ax):
	'''plot everything else before running this function, or the axis limits
	may change and the focal mechanisms may not look spherical.'''
	xaxis = ax.get_xlim()
	yaxis = ax.get_ylim()
	zaxis = ax.get_zbound()

	#get minimum and maximum bounds for each axis
	minx = min(xaxis)
	maxx = max(xaxis)
	miny = min(yaxis)
	maxy = max(yaxis)
	minz = min(zaxis)
	maxz = max(zaxis)

	
	#check if beachballs would exceed current bounds and record new bounds
	for radius, center, angles in beachball_list:
		if center[0] - radius < minx:
			minx = center[0] - radius
		if center[0] + radius > maxx:
			maxx = center[0] + radius
		if center[1] - radius < miny:
			miny = center[1] - radius
		if center[1] + radius > maxy:
			maxy = center[1] + radius
		if center[2] - radius < minz:
			minz =  center[2] - radius
		if center[2] + radius > maxz:
			maxz = center[2] + radius

	#actually set new bounds
	if xaxis[0] > xaxis[1]: #axis is inverted
		minx, maxx = maxx, minx
	ax.set_xlim(minx, maxx)
	if yaxis[0] > yaxis[1]:
		miny, maxy = maxy, miny
	ax.set_ylim(miny, maxy)
	if zaxis[0] > zaxis[1]:
		minz, maxz = maxz, minz
	ax.set_zlim(minz, maxz)


	#calculate axis lengths and normalize by longest axis
	axis_len = [maxx - minx, maxy - miny, maxz - minz]
	scale_factors = [i / max(axis_len) for i in axis_len]
	return scale_factors


def plot_focal_mechanisms(data_list, ax = None, **kwargs):
	'''kwargs:
			degrees: True or False (default True).
				If True, strike, dip, and rake angles are given
				in degrees. If False, they are given in radians.
			bottom_half: True or False (default False).
				When True, shows the bottom
				half of the sphere only, with the top half removed.
			alpha: Float between 0 and 1 (default .75).
				Sets transparancy of the focal
				mechanism sphere, with 0 being fully transparent and 1 being fully opaque
			points: Integer (default 20).
				The number of points used to make the sphere. Higher
				numbers have better resolution but make take longer to render.
			plot_planes: True or False (default True).
				If True, plot great circles that represents the focal planes.
			vector_plots: List of vectors to plot.
			vector_colors: List of colors for each vector.'''
	if ax == None:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')

	scale_factors = scale_beachballs(data_list, ax)
	for radius, center, angles in data_list:
		focal_mechanism(radius, center, angles, ax, scale_factors, **kwargs)
	if 'vector_plots' in kwargs:
		#make proxy legend
		for label, color in zip(kwargs['vector_plots'], kwargs['vector_colors']):
			ax.plot([], [], label = label, color = color)
		plt.legend()
def focal_mechanism(radius, center, angles, ax, scale_factors, degrees = True, bottom_half = False,
					alpha = .75, points = 20, plot_planes = True, vector_plots = [], vector_colors = [],
					print_vecs = False, shade = True):
	'''radius determines the size of the beach ball, center is a list of x,y,z coordinates
	for the center of the beach ball, angles is a list of the strike, dip, and slip angles,
	scale_factors is a list of the proportions to scale the x, y, and z coordinates by to compensate
	for differences in the axes, and degrees is a flag that should be set to True if the strike, dip,
	and slip angles are in degrees and False if they are in radians.
	
	Strike is 0 to 360 degrees. Dip is 0 to 90 degrees. Rake is between -180 and 180 degrees.
	'''
	
	colors = ['red', 'white', 'red', 'white']
	vecs = vectors(angles, degrees = degrees)

	quads = fm_points(angles, degrees, points)
	for color, quad in zip(colors, quads):
		x, y, z = quad


		#remove the top half of the sphere
		if bottom_half:
			#for each point, determine if the line between the point above it
			#and/or to the left (in the grid, not in xyz space), crosses
			#the xy plane.
			for i in range(points):
				for j in range(points):
					if i != 0 and z[i, j] * z[i - 1, j] < 0:
						shorten_line(x, y, z, i, j, i - 1, j)
					if j != 0 and z[i, j] * z[i, j - 1] < 0:
						shorten_line(x, y, z, i, j, i, j - 1)
					  
			x[np.where(z > 0)] = np.nan

		#multiply by radius to resize, by scale_factors to compensate for axis size differences, and add center
		#coordinates to translate to the correct location
		x = x * radius * scale_factors[0] + center[0]
		y = y * radius * scale_factors[1] + center[1]
		z = z * radius * scale_factors[2] + center[2]

		# return x, y, z
		ax.plot_surface(x, y, z, color=color, linewidth=0, alpha = alpha, shade = shade)

	if plot_planes:
		plot_circle(radius, center, vecs, ax, scale_factors, degrees = degrees)

	for vectype, c in zip(vector_plots, vector_colors):
		vec = vecs[vectype]
		plot_vector(radius, center, vec, ax, scale_factors, c)

	if print_vecs:
		print('Strike: {}°, Dip: {}°, Rake: {}°'.format(*angles))
		print_vectors(vecs)
	



