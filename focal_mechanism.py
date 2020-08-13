#!/usr/bin/env python3
# Last revised: 13/08/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import numpy as np
import pandas as pd
import os, sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import radians, sin, cos, isclose
from vector_math import shorten_line, vec_to_angles, vectors
from plotcoords import circle_arc, fm_quadrant, fm_points
from datautils import parse_file
from mpl_plots import generate_scale_factors

class Event(object):
	def __init__(self, latitude, longitude, altitude, magnitude, projection = 'equirectangular'):
		self.projection = projection
		self.magnitude = magnitude
		if projection == 'equirectangular':
			self.location = self.equirectangular_projection(latitude, longitude, altitude)
	#There may be support for projections other than equirectangular in the future
	def equirectangular_projection(self, latitude, longitude, altitude):
		return (latitude, longitude, altitude)

class FocalMechanism(Event):
	def __init__(self, latitude, longitude, altitude, magnitude, strike, dip, rake, projection = 'equirectangular', degrees = True):
		super().__init__(latitude, longitude, altitude, magnitude, projection)
		self.strike = strike
		self.dip = dip
		self.rake = rake
		self.vectors = self.calculate_vectors((strike, dip, rake), degrees = degrees)

	def print_vectors(self):
		'''Takes a dict of xyz vectors, prints the vector type, xyz vector, and plunge/bearing format.'''
		textstring = '{0}: <{1},{2},{3}>, bearing: {4}°, plunge: {5}°'
		for v in self.vectors:
			bearing, plunge = vec_to_angles(self.vectors[v])
			#shorten to two decimal places
			shortened = ['{:.2f}'.format(x) for x in [*self.vectors[v], bearing, plunge]]
			vecs_FM=textstring.format(v, *shortened)
			print(vecs_FM)
	
	def print_angles(self):
		print('Strike: {}°, Dip: {}°, Rake: {}°'.format(self.strike, self.dip, self.rake))
	
	def calculate_vectors(self, angles, degrees = True):
		if degrees:
			strike, dip, rake = [radians(x) for x in angles]
		else:
			strike, dip, rake = angles

		strike_vector = np.array([sin(strike),
								cos(strike),
								0])
		dip_vector = np.array([cos(dip)*cos(strike),
							-cos(dip)*sin(strike),
							-sin(dip)])
		rake_vector = np.array([cos(rake)*sin(strike) - sin(rake)*cos(dip)*cos(strike),
								cos(rake)*cos(strike) + sin(rake)*cos(dip)*sin(strike),
								sin(rake)*sin(dip)])
		normal_vector = np.array([sin(dip)*cos(strike),
								-sin(dip)*sin(strike),
								cos(dip)])
		null_vector = np.array([-sin(rake)*sin(strike) - cos(rake)*cos(dip)*cos(strike),
								-sin(rake)*cos(strike) + cos(rake)*cos(dip)*sin(strike),
								cos(rake)*sin(dip)])

		p_vector = np.array([sin(dip)*cos(strike) - cos(rake)*sin(strike) + sin(rake)*cos(dip)*cos(strike),
							-sin(dip)*sin(strike) - cos(rake)*cos(strike) - sin(rake)*cos(dip)*sin(strike),
							cos(dip) - sin(rake)*sin(dip)])

		t_vector = np.array([sin(dip)*cos(strike) + cos(rake)*sin(strike) - sin(rake)*cos(dip)*cos(strike),
							-sin(dip)*sin(strike) + cos(rake)*cos(strike) + sin(rake)*cos(dip)*sin(strike),
							cos(dip) + sin(rake)*sin(dip)])
		#sanity checks

		#normal vector should be the cross product of dip vector and strike vector
		norm_correct = np.isclose(normal_vector, np.cross(dip_vector, strike_vector))   
		assert(norm_correct.all())
		
		#rake vector should be cos(rake) * strike_vector - sin(rake) * dip_vector
		rake_correct = np.isclose(rake_vector, cos(rake) * strike_vector - sin(rake) * dip_vector)                            
		assert(rake_correct.all())
		
		#null vector should be the cross product of normal vector and rake vector
		null_correct = np.isclose(null_vector, np.cross(normal_vector, rake_vector))
		assert(null_correct.all())

		#p should be normal - rake
		p_correct = np.isclose(p_vector, normal_vector - rake_vector)
		assert(p_correct.all())
		
		#t should be normal + rake
		t_correct = np.isclose(t_vector, normal_vector + rake_vector)
		assert(t_correct.all())

		#normalize p and t so they have a length of 1
		p_vector = p_vector / np.linalg.norm(p_vector)
		t_vector = t_vector / np.linalg.norm(t_vector)

		#check if null, p, and t are pointing downward. If not, reverse them.
		if null_vector[2] > 0:
			null_vector = null_vector * -1
		if p_vector[2] > 0:
			p_vector = p_vector * -1
		if t_vector[2] > 0:
			t_vector = t_vector * -1

		return {'strike': strike_vector,
				'dip' : dip_vector,
				'rake' : rake_vector,
				'normal' : normal_vector,
				'B': null_vector,
				'P': p_vector,
				'T': t_vector}


def plot_circle(focalmechanism, ax, scale_factors, fault_color = 'black', auxiliary_color = 'blue',
				degrees = True):
	vecs = focalmechanism.vectors
	center = focalmechanism.location
	radius = focalmechanism.magnitude
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




def plot_focal_mechanisms(data_list, ax = None, degrees = True, **kwargs):
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
	focalmechanisms = []
	for magnitude, location, angles in data_list:
		focalmechanisms.append(FocalMechanism(*location, magnitude, *angles, degrees = degrees))

	scale_factors = generate_scale_factors(focalmechanisms, ax)
	for fm in focalmechanisms:
		focal_mechanism(fm, ax, scale_factors, **kwargs)
	if 'vector_plots' in kwargs:
		#make proxy legend
		for label, color in zip(kwargs['vector_plots'], kwargs['vector_colors']):
			ax.plot([], [], label = label, color = color)
		plt.legend()
def focal_mechanism(fm, ax, scale_factors, bottom_half = False,
					alpha = .75, points = 20, plot_planes = True, vector_plots = [], vector_colors = [],
					print_vecs = False, shade = True):
	'''radius determines the size of the beach ball, center is a list of x,y,z coordinates
	for the center of the beach ball, angles is a list of the strike, dip, and slip angles,
	scale_factors is a list of the proportions to scale the x, y, and z coordinates by to compensate
	for differences in the axes, and degrees is a flag that should be set to True if the strike, dip,
	and slip angles are in degrees and False if they are in radians.
	
	Strike is 0 to 360 degrees. Dip is 0 to 90 degrees. Rake is between -180 and 180 degrees.
	'''
	radius = fm.magnitude
	center = fm.location
	colors, quads = fm_points(fm, points)
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
		plot_circle(fm, ax, scale_factors, degrees = degrees)

	for vectype, c in zip(vector_plots, vector_colors):
		vec = fm.vectors[vectype]
		plot_vector(radius, center, vec, ax, scale_factors, c)
	



