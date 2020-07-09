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
    #generate points for quarter-sphere
    v = np.linspace(0, np.pi, points)
    u = np.linspace(border, border + np.pi/2, points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    #combine into coordinate matrix so rotation can be applied
    coordinate_matrix = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    #apply rotations to matrix

    if degrees:
        offset = 90
    else:
        offset = np.pi / 2
    slip_rotation = Rotation.from_euler('x', angles[2], degrees = degrees)
    dip_rotation = Rotation.from_euler('y', angles[1] - offset, degrees = degrees)
    strike_rotation = Rotation.from_euler('z', -angles[0], degrees = degrees)
    slip_rotated = slip_rotation.apply(coordinate_matrix)
    dip_rotated = dip_rotation.apply(slip_rotated)
    strike_rotated = strike_rotation.apply(dip_rotated)




    #separate x, y, and z matrices
    x = strike_rotated[:, 0]
    y = strike_rotated[:, 1]
    z = strike_rotated[:, 2]

    #unflatten
    x = x.reshape(points, points)
    y = y.reshape(points, points)
    z = z.reshape(points, points)

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
    x = np.flip(X, 0)
    y = np.flip(np.flip(Y, 0), 1)
    z = np.flip(np.flip(Z, 0), 1)
    return x, y, z

def fm_points(angles, degrees, points):
    borders = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    quads = []
    v = np.linspace(0, np.pi, points)
    for border in borders:
        #generate points for quarter-sphere
        u = np.linspace(border, border + np.pi/2, points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        #combine into coordinate matrix so rotation can be applied
        coordinate_matrix = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        #apply rotations to matrix

        if degrees:
            offset = 90
        else:
            offset = np.pi / 2
        slip_rotation = Rotation.from_euler('x', angles[2], degrees = degrees)
        dip_rotation = Rotation.from_euler('y', angles[1] - offset, degrees = degrees)
        strike_rotation = Rotation.from_euler('z', -angles[0], degrees = degrees)
        slip_rotated = slip_rotation.apply(coordinate_matrix)
        dip_rotated = dip_rotation.apply(slip_rotated)
        strike_rotated = strike_rotation.apply(dip_rotated)




        #separate x, y, and z matrices
        x = strike_rotated[:, 0]
        y = strike_rotated[:, 1]
        z = strike_rotated[:, 2]

        #unflatten
        x = x.reshape(points, points)
        y = y.reshape(points, points)
        z = z.reshape(points, points)
        quads.append((x, y, z))
    return quads

def new_fm_points(angles, degrees, points):
    borders = [0, 3* np.pi / 2, np.pi, np.pi / 2,]
    quads = []
    for border in borders:
        quads.append(fm_quadrant(border, angles, degrees, points))
    return quads
