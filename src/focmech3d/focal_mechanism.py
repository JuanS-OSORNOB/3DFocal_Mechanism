#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import os, sys
from math import radians, sin, cos, isclose, degrees

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from focmech3d.vector_math import vec_to_angles, remove_top, angle_between
from focmech3d.plotcoords import fm_quadrant, fm_points, translate_and_scale
from focmech3d.mpl_plots import plot_circle, plot_vector, generate_scale_factors, plot_focal_mechanism, get_data_limits

class Event(object):
	def __init__(self, longitude, latitude, altitude, magnitude, *other_params, projection = 'equirectangular', rad_function = lambda x: x, colnames = [],
	invert_z = False):
		'''Rad function is a function that takes the magnitude and turns it into a focal mechanism radius. By default, the magnitude is simply the radius.'''
		self.projection = projection
		self.magnitude = magnitude
		self.radius = rad_function(magnitude)

		if invert_z:
			altitude *= -1
		if projection == 'equirectangular':
			self.location = self.equirectangular_projection(longitude, latitude, altitude)
		if colnames:
			self.other_params_dict = dict(zip(colnames, other_params))
		elif other_params:
			self.other_params = other_params
	#There may be support for projections other than equirectangular in the future
	def equirectangular_projection(self, longitude, latitude, altitude):
		return (longitude, latitude, altitude)

class FocalMechanism(Event):
	'''This is an earthquake event at a particular location in xyz space, with a magnitude and strike, dip, and rake angles.
	A set of vectors that characterizes the focal mechanism is derived from the strike, dip, and rake angles.
	Currently, longitude, latitude, and altitude are converted to x, y, and z, respectively, as in an equirectangular map
	projection. The positive x-axis points east, the positive y-axis points north, and the positive z-axis points up. Set invert_z to True 
	if the z-axis numbers represent depth rather than altitude (e.g. an event 30km deep has a z-value of 30 rather than -30 in the input data.)
	The magnitude of the earthquake is used to determine the radius of the focal mechanism plot. The strike vector represents the direction
	of the fault in the xy-plane. It is defined such that the dip is downward and to the right 
	(i.e. the hanging wall is on the right) when looking in the direction of the strike vector. The strike angle is the angle between 
	the y-axis (north) and the strike vector, measured in the clockwise direction. The dip vector is perpendicular to the strike
	vector and defines how the fault plane angles down from the xy-plane. The dip angle is measured from the vector in the xy-plane that 
	is 90 degrees clockwise from the strike vector. 0 degrees represents a dip vector that is in the xy-plane, while 90 degrees represents a
	dip vector that points straight down. The rake vector is in the plane defined by the strike and dip vectors and represents the 
	direction of movement of the hanging wall. 0 degrees of rake represents movement in the direction of the strike, while -90 degrees
	represents movement in the direction of the dip.
	
	Strike is 0 to 360 degrees. Dip is 0 to 90 degrees. Rake is between -180 and 180 degrees.'''

	def __init__(self, longitude, latitude, altitude, magnitude, strike, dip, rake, *other_params, projection = 'equirectangular', in_degrees = True,
				rad_function = lambda x: x, colnames = [], invert_z = False):
		super().__init__(longitude, latitude, altitude, magnitude, projection = projection, rad_function = rad_function, invert_z = invert_z)
		if in_degrees:
			if strike < 0 or strike > 360:
				raise Exception('Expected strike between 0 and 360 degrees; got {}'.format(strike))
			if dip < 0 or dip > 90:
				raise Exception('Expected dip between 0 and 90 degrees; got {}'.format(dip))
			if rake < -180 or rake > 180:
				raise Exception('Expected rake between -180 and 180 degrees; got {}'.format(rake))
			strike, dip, rake = [radians(angle) for angle in (strike, dip, rake)]
		else:
			if strike < 0 or strike > 2*np.pi:
				raise Exception('Expected strike between 0 and 2*pi radians; got {}'.format(strike))
			if dip < 0 or dip > np.pi/2:
				raise Exception('Expected dip between 0 and pi/2 degrees; got {}'.format(dip))
			if rake < -np.pi or rake > np.pi:
				raise Exception('Expected rake between -pi and pi degrees; got {}'.format(rake))
		self.strike = strike
		self.dip = dip
		self.rake = rake
		self.vectors = self.calculate_vectors()
		if colnames:
			self.other_params_dict = dict(zip(colnames, other_params))
		elif other_params:
			self.other_params = other_params

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
		print('Strike: {}°, Dip: {}°, Rake: {}°'.format(degrees(self.strike), degrees(self.dip), degrees(self.rake)))
	
	def calculate_vectors(self):
		strike = self.strike
		dip = self.dip
		rake = self.rake

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
	
	def aux_plane(self):
		#still under testing; may not be correct
		'''Returns strike, dip, and rake vectors for auxiliary plane.'''
		strike = np.cross(np.array([0, 0, 1]), self.vectors['rake'])
		dip = np.cross(strike, self.vectors['rake'])
		return strike, dip, self.vectors['normal']
		
	def aux_plane_angles(self):
		#still under testing; may not be correct
		'''Returns angles (in degrees) for the auxiliary strike, dip, and rake'''
		strike, dip, rake = self.aux_plane()
		strike_angle, _ = vec_to_angles(strike)
		_, dip_angle = vec_to_angles(dip)
		rake_angle = degrees(angle_between(strike, rake))
		return strike_angle, dip_angle, rake_angle


def plot_focal_mechanisms(data_list, ax = None, in_degrees = True, in_fms = False, **kwargs):
	'''kwargs:
			in_degrees: True or False (default True).
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
	if in_fms == False:
		for magnitude, location, angles in data_list:
			focalmechanisms.append(FocalMechanism(*location, magnitude, *angles, in_degrees = in_degrees))
	else:
		focalmechanisms = data_list

	scale_factors = generate_scale_factors(focalmechanisms, ax)
	for fm in focalmechanisms:
		plot_focal_mechanism(fm, ax, scale_factors, **kwargs)
	if 'vector_plots' in kwargs:
		#make proxy legend
		for label, color in zip(kwargs['vector_plots'], kwargs['vector_colors']):
			ax.plot([], [], label = label, color = color)
		plt.legend()
	return ax
