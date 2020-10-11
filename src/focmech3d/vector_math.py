#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import numpy as np
from math import isclose, radians, sin, cos, atan2, asin

def remove_top(coords):
	'''Takes a set of 3D coordinates. Removes everything above the xy plane.'''
	x, y, z = coords
	#for each point, determine if the line between the point above it
	#and/or to the left (in the grid, not in xyz space), crosses
	#the xy plane.
	for i in range(len(x)):
		for j in range(len(x)):
			if i != 0 and z[i, j] * z[i - 1, j] < 0:
				shorten_line(x, y, z, i, j, i - 1, j)
			if j != 0 and z[i, j] * z[i, j - 1] < 0:
				shorten_line(x, y, z, i, j, i, j - 1)
				
	x[np.where(z > 0)] = np.nan
	return np.array([x, y, z])

def circle_angle(axis1, axis2, vec):
	'''Returns the angle(in radians) between vec and axis1. Uses axis2 for directionality
	(i.e. axis2 is at pi/2 rather than 3pi/2).'''
	angle = angle_between(axis1, vec)
	chirality = angle_between(axis2, vec)
	if chirality > np.pi/2:
		return 2 * np.pi - angle
	return angle

def translate_rotate_point(x, y, angle, center):
	'''Moves a point (x, y) so that the new point is the same with respect
	to the origin as the point (x, y) was with respect to the point 'center'.
	Then rotate the new point around the origin by 'angle', counterclockwise from the
	x-axis.'''
	x, y = x - center[0], y - center[1]
	newx = x * cos(angle) - y * sin(angle)
	newy = x * sin(angle) + y * cos(angle)
	return newx, newy

def normalize_vector(vector):
	#cast to float
	vector = vector.astype(float)
	return vector / np.linalg.norm(vector)

def angle_between(vec1, vec2):
	vec1 = normalize_vector(vec1)
	vec2 = normalize_vector(vec2)
	dotprod = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
	return np.arccos(dotprod)

def vectors(angles, degrees = True):
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



def shorten_line(x, y, z, i, j, i2, j2):
	'''shorten line between <x[i,j], y[i,j], z[i,j]>
	and <x[i2,j2], y[i2,j2], z[i2,j2]> so that the point above the xy
	plane lies on it'''
	if z[i, j] < 0:
		#if z[i, j] is the smaller of the two, switch indices so it's larger
		i, j, i2, j2 = i2, j2, i, j
	#now <x[i,j], y[i,j], z[i,j]> is the point to be moved

	#calculate fraction of line to remove in each dimension
	zfrac = z[i, j] / (z[i, j] - z[i2, j2])
	xdist = zfrac * (x[i, j] - x[i2, j2])
	ydist = zfrac * (y[i, j] - y[i2, j2])

	z[i, j] = 0
	x[i, j] = x[i, j] - xdist
	y[i, j] = y[i, j] - ydist


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

def get_data_limits(focalmechanisms):
	'''Returns the largest and smallest x, y, and z values of the locations of the focal mechanisms, +/- the radius.'''
	centers = np.array([fm.location for fm in focalmechanisms]).T
	radii = np.array([fm.radius for fm in focalmechanisms])
	X, Y, Z = centers
	minx = min(X - radii)
	maxx = max(X + radii)
	miny = min(Y - radii)
	maxy = max(Y + radii)
	minz = min(Z - radii)
	maxz = max(Z + radii)
	return minx, maxx, miny, maxy, minz, maxz