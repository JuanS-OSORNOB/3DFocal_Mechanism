import numpy as np
from math import isclose, radians, sin, cos
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.testing.compare import compare_images

def circle_arc(axis1, axis2, start_angle, end_angle, center = [0, 0, 0], scale_factors = [1, 1, 1], radius = 1, points = 50):
	'''Generates xyz coordinates for an arc of a circle. The vectors axis1
	and axis2 must be perpendicular and in the plane of the circle.
	The start_angle and end_angle determine which part of the circle
	is included in the arc, with an angle of 0 corresponding to axis1 and
	an angle of pi/2 corresponding to axis2. Angles are in radians.'''

	if start_angle > end_angle:
		end_angle += 2 * np.pi
	
	if not isclose(np.dot(axis1, axis2), 0, abs_tol = 1e-09):
		raise Exception('Axes must be perpendicular.')
	angle_points = np.linspace(start_angle, end_angle, points)
	x = axis1[0] * np.cos(angle_points) + axis2[0] * np.sin(angle_points)
	y = axis1[1] * np.cos(angle_points) + axis2[1] * np.sin(angle_points)
	z = axis1[2] * np.cos(angle_points) + axis2[2] * np.sin(angle_points)

	x = x * radius * scale_factors[0] + center[0]
	y = y * radius * scale_factors[1] + center[1]
	z = z * radius * scale_factors[2] + center[2]

	return np.array([x, y, z])

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

def fm_quadrant(border, angles, degrees, points):
    elevs = np.linspace(np.pi/2, -np.pi/2, points)
    azims = np.linspace(border + np.pi/2, border, points)
    vecs = vectors(angles, degrees = degrees)
    rake = np.array([vecs['rake']]).T
    normal = np.array([vecs['normal']]).T
    null = np.array([vecs['B']]).T
    X1 = []
    Y1 = []
    Z1 = []
    for azim in azims:
        x, y, z = (np.cos(azim) * rake + np.sin(azim) * normal) * np.cos(elevs) + np.sin(elevs)*null
        X1.append(x)
        Y1.append(y)
        Z1.append(z)
    X = np.array(X1)
    Y = np.array(Y1)
    Z = np.array(Z1)
    x = X
    y = Y
    z = Z
    return x, y, z

def fm_points(angles, degrees, points):
    borders = [0, np.pi / 2, np.pi, 3 * np.pi / 2,]
    quads = []
    for border in borders:
        quads.append(fm_quadrant(border, angles, degrees, points))
    return quads

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