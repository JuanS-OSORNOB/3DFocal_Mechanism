import numpy as np
from math import isclose

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


def fm_quadrant(border, focalmechanism, points):
    elevs = np.linspace(np.pi/2, -np.pi/2, points)
    azims = np.linspace(border + np.pi/2, border, points)
    vecs = focalmechanism.vectors
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

def fm_points(focalmechanism, points):
    borders = [0, np.pi / 2, np.pi, 3 * np.pi / 2,]	
    colors = ['red', 'white', 'red', 'white']
    quads = []
    for border in borders:
        quads.append(fm_quadrant(border, focalmechanism, points))
    return colors, quads