#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import numpy as np
from math import isclose

'''Functions in this module are intended to return relative coordinates (i.e. centered at the origin, with a radius of 1),
except for translate_and_scale, which can be used to move and stretch the coordinates so they are in absolute form.'''

def circle_arc(axis1, axis2, start_angle, end_angle, points = 50):
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

	return np.array([x, y, z])


def fm_quadrant(border, focalmechanism, points):
    elevs = np.linspace(np.pi/2, -np.pi/2, points)
    azims = np.linspace(border + np.pi/2, border, points)
    vecs = focalmechanism.vectors
    rake = np.array([vecs['rake']]).T
    normal = np.array([vecs['normal']]).T
    null = np.array([vecs['B']]).T
    X = []
    Y = []
    Z = []
    for azim in azims:
        x, y, z = (np.cos(azim) * rake + np.sin(azim) * normal) * np.cos(elevs) + np.sin(elevs)*null
        X.append(x)
        Y.append(y)
        Z.append(z)
    return np.array(X), np.array(Y), np.array(Z)

def fm_points(focalmechanism, points):
    borders = [0, np.pi / 2, np.pi, 3 * np.pi / 2,]	
    colors = ['red', 'white', 'red', 'white']
    quads = []
    for border in borders:
        quads.append(fm_quadrant(border, focalmechanism, points))
    return colors, quads

def translate_and_scale(coords, center_vector, scale_factors = None):
    '''Takes a set of coordinates in the form [[x1, x2, ..., xn], [y1, y2, ..., yn]]
    or [[x1, x2, ..., xn], [y1, y2, ..., yn], [z1, z2, ..., zn]] and scales them according to scale_factors, 
    then translates them from their current location to location + center_vector. Generally, this will be used for coordinates
    centered at the origin, since scaling will give unexpected results if the coordinates are not
    centered at the origin. In this case, the new location will be at center_vector. 
    However, this function can be used for translation only, if no value is given for scale_factors.'''
    if len(coords) == 2:
        x, y = coords
        cx, cy = center_vector
        if scale_factors:
            sx, sy = scale_factors
        else:
            sx, sy = 1, 1
    elif len(coords) == 3:
        x, y, z = coords
        cx, cy, cz = center_vector
        if scale_factors:
            sx, sy, sz = scale_factors
        else:
            sx, sy, sz = 1, 1, 1
    else:
        raise Exception('2 or 3 sets of coordinates expected; got {}'.format(len(coords)))
    x = translate_and_scale_single(x, cx, sx)
    y = translate_and_scale_single(y, cy, sy)
    if len(coords) == 3:
        z = translate_and_scale_single(z, cz, sz)
        return x, y, z
    return x, y

def translate_and_scale_single(coords, location, scale_factor = 1):
    '''Takes a list of coordinates and multiplies them by scale_factor, then adds location.'''
    coords = np.array(coords)
    coords = coords * scale_factor + location
    return coords