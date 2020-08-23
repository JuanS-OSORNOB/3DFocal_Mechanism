#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
from math import isclose, atan2, sqrt
import os, sys

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from focmech3d.focal_mechanism import plot_focal_mechanisms, plot_vector
from focmech3d.vector_math import translate_rotate_point, normalize_vector, angle_between, circle_angle, vectors
from focmech3d.plotcoords import circle_arc
from focmech3d.mpl_plots import generate_scale_factors

def lambert_projection(point, center_point, new_x_axis, new_y_axis):
	'''Generates the Lambert azimuthal equal-area projection of a point on a sphere
	centered at the origin. The point is projected onto a plane tangent to
	the sphere at center_point. new_y_axis and new_x_axis determine the angle at
	which the plane is viewed. '''

	#displacement is the vector from the center_point to the point to be projected
	displacement = point - center_point
	magnitude = np.linalg.norm(displacement)

	#project the displacement vector to create a vector pointing in the same direction in the projection plane
	xproj = np.dot(displacement, new_x_axis)
	yproj = np.dot(displacement, new_y_axis)
	xy_vec = np.array([xproj, yproj])

	#divide by the norm to make a unit vector and then multiply by magnitude so it is the same length
	#as the displacement vector
	xy_vec = xy_vec * magnitude / np.linalg.norm(xy_vec)

	return xy_vec

def xyz_to_lambert(X, Y, Z, projection_point, new_x_axis, new_y_axis):
	xy_coords = []
	for x, y, z in zip(X, Y, Z):
		point = np.array([x, y, z])
		xy_coords.append(lambert_projection(point, projection_point, new_x_axis, new_y_axis))
	return xy_coords

def lambert(projection_point, new_y_axis, vecs):
	'''Generates a Lambert azimuthal equal-area projection of the near hemisphere of a focal mechanism
	from an arbitrary projection_point on the sphere. Points on the sphere are projected onto a plane
	tangent to the sphere at projection_point, which becomes the origin of the plane. new_y_axis determines
	the orientation of the plane when viewed. Essentially, this function shows an approximation of what
	a focal mechanism sphere would look like while looking directly at projection_point, with new_y_axis being 'up'.
	vecs is a set of characteristic vectors for a focal mechanism.

	The outer circle of the projection will be defined by the plane through the center of the sphere,
	perpendicular to the projection_point vector. If the projection point is the north pole, this circle
	is the equator. We also need to determine and plot the great circles that represent the nodal planes of the focal mechanism.
	Since we are only plotting the near hemisphere, we need to determine what part of the circles are in the hemisphere.
	For this, it is useful to find the intersection of the equator and each great circle.
	The normal vectors for the planes of each circle are known (they are the 'normal' and 'rake' vectors).
	The intersection point must be in the plane of the equator (therefore perpendicular to the projection point)
	and in the plane of the great circle (therefore perpendicular to the normal vector for the plane.

	In most cases, the hemisphere will be divided into four sections. Each section is defined by three intersecting arcs:
	one from each nodal plane circle and one from the outer equitorial circle.

	'''
	
	#note: figure out what to do if projection point is the same as normal vectors

	projection_point = normalize_vector(projection_point)
	new_y_axis = normalize_vector(new_y_axis)

	#determine an x-axis (perpendicular to both the projection_point vector and the new_y_axis vector.
	new_x_axis = np.cross(new_y_axis, projection_point)
	if not isclose(np.linalg.norm(new_x_axis), 1):
		raise Exception('Y-axis vector must be perpendicular to pole vector.')

	#Find intersection point of fault plane circle and projection equator 
	fp_int = np.cross(projection_point, vecs['normal'])
	fp_int = normalize_vector(fp_int)

	#Circle generator takes two perpendicular vectors in plane of circle. Find orthogonal vector with cross product
	fp_orth = np.cross(vecs['normal'], fp_int)
	#We want the orthogonal vector to be in the near hemisphere. Reverse if not.
	if np.dot(fp_orth, projection_point) < 0:
		fp_orth *= -1

	#find intersection and orthogonal vector for auxiliary plane
	ap_int = np.cross(projection_point, vecs['rake'])
	ap_int = normalize_vector(ap_int)
	#Check if ap_int is clockwise from fp_int (in other words, cross product of ap_int and fp_int should be toward viewer)
	crossprod = np.cross(ap_int, fp_int)
	if np.dot(crossprod, projection_point) < 0:
		ap_int *= -1
	ap_orth = np.cross(vecs['rake'], ap_int)
	if np.dot(ap_orth, projection_point) < 0:
		ap_orth *= -1
	
	#use t-vector to determine which quadrants to fill
	tension = vecs['T']
	#make sure T is in near-hemisphere
	if np.dot(tension, projection_point) < 0:
		tension *= -1

	#determine angle from new_y_axis that each intersection point is
	angles = [circle_angle(new_y_axis, new_x_axis, vec) for vec in [fp_int, ap_int, -fp_int, -ap_int]]
	fp_angle, ap_angle, neg_fp_angle, neg_ap_angle = angles
			  
	#the quadrant that T is in can be determined by which of fp_int and negative fp_int it is closer to,
	#and which of ap_int and negative ap_int it is closer to.

	fp_ap = False
	if angle_between(fp_int, tension) < angle_between(-fp_int, tension):
		if angle_between(ap_int, tension) < angle_between(-ap_int, tension):
			#tension is in fp-ap quadrant, so fp-ap quadrant and neg_fp-neg_ap quadrant are filled.
			fp_ap = True
		else:
			#tension is in neg_ap-fp quadrant, so neg_ap-fp and ap-neg_fp quadrant are filled
			fp_ap = False
	elif angle_between(ap_int, tension) < angle_between(-ap_int, tension):
		#tension is in ap-neg_fp quadrant, so neg_ap-fp and ap-neg_fp quadrant are filled
		fp_ap = False
	else:
		#tension is in neg_fp-neg_ap quadrant, so fp-ap quadrant and neg_fp-neg_ap quadrant are filled.
		fp_ap = True

	fp_arc = circle_arc(fp_int, fp_orth, 0, np.pi)[:, ::-1]
	ap_arc = circle_arc(ap_int, ap_orth, 0, np.pi)

	if fp_ap:
		arc1 = circle_arc(new_y_axis, new_x_axis, fp_angle, ap_angle)
		arc2 = circle_arc(new_y_axis, new_x_axis, neg_fp_angle, neg_ap_angle)
		filled_area = np.concatenate([arc1, ap_arc, arc2[:, ::-1], fp_arc], axis = 1)
	else:
		arc1 = circle_arc(new_y_axis, new_x_axis, ap_angle, neg_fp_angle)
		arc2 = circle_arc(new_y_axis, new_x_axis, neg_ap_angle, fp_angle)[:, ::-1]
		filled_area = np.concatenate([arc1, fp_arc, arc2, ap_arc[:, ::-1]], axis = 1)

	outer_circle = circle_arc(new_y_axis, new_x_axis, 0, 2 * np.pi)

	coords = []
	for arc in [outer_circle, filled_area]:
		coords.append(xyz_to_lambert(arc[0, :], arc[1, :], arc[2, :], projection_point, new_x_axis, new_y_axis))

	return coords

def profile_view(x1, y1, x2, y2, width, depth):
	'''A profile is a view of a slice of terrain and associated events. It consists of a right rectangular prism whose upper face is 
	on the surface (at depth 0). The surface rectangle is defined by the variables x1, y1, x2, y2, and width. (x1, y1) and (x2, y2) are 
	points on the surface (in the XY plane). The line segment from (x1, y1) to (x2, y2) is the midline of the surface rectangle, which therefore
	has two parallel sides whose midpoints are (x1, y1) and (x2, y2) and whose length is 'width' units. The lower face of the prism is
	'depth' units below the surface and has the same xy coordinates as the upper face. The view is perpendicular to this midline, so that 
	the viewing plane has the left side corresponding to (x1, y1), the right side corresponding to (x2, y2), the bottom corresponding to (-depth) 
	and the top corresponding to the surface. 
	
	[Describe what this function actually does]'''

	if x1 == x2 and y1 == y2:
		raise Exception('Endpoints must not be the same point.')

	#vector in direction of midline, from (x1, y1) to (x2, y2)
	vec1 = np.array([x2 - x1, y2 - y1, 0])
	#viewing plane is vertical so another vector is z unit vector
	vec2 = np.array([0, 0, 1])
	#vector normal to both of these is normal to the plane (points towards viewer)
	norm_vec = np.cross(vec1, vec2)
	#normal vector should be in XY plane, so
	assert(norm_vec[2] == 0)
	norm_vec = norm_vec / np.linalg.norm(norm_vec)
	
	#find limits of bounding box
	corner1 = np.array([x1, y1, 0]) - width/2 * norm_vec #lower left corner after rotation
	corner2 = corner1 + width * norm_vec #lower right corner after rotation
	corner3 = np.array([x2, y2, 0]) + width/2 * norm_vec #upper right corner after rotation
	corner4 = corner3 - width * norm_vec #upper left corner after rotation
	original_corners = (corner1, corner2, corner3, corner4)
	
	#find center of bounding box (midpoint of diagonal)
	centerx = (corner1[0] + corner3[0]) / 2
	centery = (corner1[1] + corner3[1]) / 2
	translate = np.array([centerx, centery])

	#find rotation angle of normal vector clockwise from x-axis
	theta = -atan2(norm_vec[1], norm_vec[0])
	
	#recenter and rotate corners
	corners = [translate_rotate_point(corner[0], corner[1], theta, translate) for corner in original_corners]
	corner1, corner2, corner3, corner4 = corners
	#establish bounds
	xmin, ymin = corner1
	xmax, ymax = corner3

	assert(xmin <= xmax)
	assert(ymin <= ymax)
	assert(isclose(xmax - xmin, width))

	#the z-value of a focal mechanism is expected to be a negative number, so depth
	#should start at a negative number and go to 0
	#or maybe modify this later to specify both ends of depth range

	if depth > 0:
		depth *= -1
	zmin = depth
	zmax = 0

	bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

	return original_corners, bounds, theta, translate, norm_vec

def plot_bounding_box(ax, A, Aprime, corners, depth):
	'''Plots the profile bounding box in 3D (see profile_view for more details about profiles).'''
	x_A, y_A = A
	x_Aprime, y_Aprime = Aprime
	x_values=[x_A, x_Aprime]
	y_values=[y_A, y_Aprime]
	ax.plot(x_values, y_values, c='purple')
	ax.scatter(x_A, y_A, c='k')
	ax.scatter(x_Aprime, y_Aprime, c='k')
	ax.text(x_A, y_A, 0, 'A', size=15)
	ax.text(x_Aprime, y_Aprime, 0, "A'", size=15)

	corner_labels = ['V1', 'V2', 'V4', 'V3']
	for v, label in zip(corners, corner_labels):
		ax.scatter(*v, c = 'k')
		ax.text(*v, label, size = 10)
	
	#create lower corner coordinates by subtracting the depth from the z value
	lower_corners = [v + np.array([0, 0, -depth]) for v in corners]

	#plot vertical lines between pairs of upper and lower corners
	for V_upper, V_lower in zip(corners, lower_corners):
		ax.plot(*zip(V_upper, V_lower), c = 'k', linestyle = 'dotted')

	#add first value of upper and lower rectangle to list so that the line forms a complete box
	five_corners = list(corners) + [corners[0]]
	five_lower_corners = lower_corners + [lower_corners[0]]

	#plot upper and lower rectangle
	ax.plot(*zip(*five_corners), c='k', linestyle='dashed')
	ax.plot(*zip(*five_lower_corners), c='k', linestyle='dashed')
	return x_A, y_A, x_Aprime, y_Aprime

def in_bounds(data_list, bounds, center, theta, rotated = True):
	'''Determines if the center points in data_list are within the bounds of a 3D bounding box whose upper face has center at 'center',
	 whose y-axis is rotated theta radians clockwise from the absolute y-axis, and which has the bounds listed in 'bounds'.
	 
	 Returns a list of events from data_list that are within the bounding box. If rotated = False, returns the events as-is. If rotated
	 = True, repackages the event with the new x and y values. Lower rotated x-values will end up towards the back of the profile view;
	 lower rotated y-values will end up towards the left.'''

	in_bounds_list = []

	xmin, xmax, ymin, ymax, zmin, zmax = bounds
	for event in data_list:
		x, y, z = event[1]
		newx, newy = translate_rotate_point(x, y, theta, center)
		if xmin <= newx <= xmax and ymin <= newy <= ymax and zmin <= z <= zmax:
			if rotated == True:
				event = (newx, newy, event)
			in_bounds_list.append(event)
	return in_bounds_list

def arrayize(points_list):
	'''Turns a list or tuple of xy pairs or xyz triples into two or three arrays.'''
	vecs = []
	for v in zip(*points_list):
		vecs.append(np.array(v))
	return vecs

def plot_lambert(ax, center, radius, scale_factors, zorder, *args):
	zorder += 1
	outer_circle, filled_area = lambert(*args)
	X, Y = arrayize(outer_circle)
	sc_x, sc_y = scale_factors
	center_x, center_y = center
	ax.plot(X * sc_x * radius + center_x, Y * sc_y * radius + center[1], color = 'black', zorder = zorder * 2) 
	ax.fill(X * sc_x * radius + center_x, Y * sc_y * radius + center[1], color = 'white', zorder = zorder * 2 - 1)
	X, Y = arrayize(filled_area) 
	ax.plot(X * sc_x * radius + center_x, Y * sc_y * radius + center[1], color = 'black', zorder = zorder * 2) 
	ax.fill(X * sc_x * radius + center_x, Y * sc_y * radius + center[1], color = 'red', zorder = zorder * 2 - 1)

def pltcolor(lst):
	cols=[]
	for l in lst:
		if l>=-30 and l<0:
			cols.append('red')
		elif l>=-70 and l<-30:
			cols.append('yellow')
		elif l>=-120 and l<-70:
			cols.append('springgreen')
		elif l>=-180 and l<-120:
			cols.append('dodgerblue')
		else:
			cols.append('b')
	return cols

def pltsize(lst):
	init=8
	size=[]
	for l in lst:
		if l>=0.0 and l<=3.0:
			size.append(init)
		elif l>3.0 and l<=4.0:
			size.append(init**2)
		elif l>4.0 and l<=5.0:
			size.append(init**2.5)
		elif l>5.0 and l<=6.0:
			size.append(init**3)
		else:
			size.append(init**3.5)
	return size


def gcs_degree_to_km(degree):
	return 111 * degree

def latlong_to_km(coords):
	return [gcs_degree_to_km(coords[0]), gcs_degree_to_km(coords[1]), coords[2]]
	

def plot_profile(FM_data_list, events_list, x1, y1, x2, y2, width, depth, fm_size = 1, depth_mag=True, in_degrees = True, verbose = True, **kwargs):
	if in_degrees:
		x1 = gcs_degree_to_km(x1)
		x2 = gcs_degree_to_km(x2)
		y1 = gcs_degree_to_km(y1)
		y2 = gcs_degree_to_km(y2)
		width = gcs_degree_to_km(width)
		for data in FM_data_list:
			data[1] = latlong_to_km(data[1])
		for data in events_list:
			data[1] = latlong_to_km(data[1])


	#Beachball projection
	original_corners, bounds, theta, center, norm_vec = profile_view(x1, y1, x2, y2, width, depth)
	in_bounds_list = in_bounds(FM_data_list, bounds, center, theta)
	if verbose:
		print('Total FM in bounds:', len(in_bounds_list))
	fig=plt.figure(dpi=220, tight_layout=True)
	ax=fig.add_subplot(111)
	plt.grid(which='major', axis='x', linestyle='--', alpha=0.5)
	_, _, ymin, ymax, zmin, zmax = bounds
	ax.set_xlim(0, ymax-ymin)
	ax.set_ylim(zmin, zmax)
	ax.set_xlabel('Relative profile distance (km)')
	ax.set_ylabel('Depth (km)')
	ax.set_aspect('equal')
	#resize focal mechanisms based on fm_size -- allows the user to specify the radius (in km) of a magnitude 1 focal mechanism.
	scale_factors = [fm_size, fm_size] 
	if 'Title' in kwargs:
		ax.set_title(kwargs['Title'], fontsize=13)
	in_bounds_list.sort() #first value was just for sorting back to front
	for i in range(len(in_bounds_list)):
		_, x, event = in_bounds_list[i]
		radius, center, angles = event
		vecs = vectors(angles)
		plot_lambert(ax, (x-ymin, center[2]), radius, scale_factors, i, norm_vec, np.array([0, 0, 1]), vecs)

	#Point profile
	Event_list=in_bounds(events_list, bounds, center, theta)
	if verbose:
		print('Total events:', len(events_list), '\nTotal events in bounds:', len(Event_list))
	depth_list, mag_list=[], []
	for i in range(0, len(Event_list)):
		newx, newy, event=Event_list[i]
		mag, center=event
		lon,lat, depth=center
		depth_list.append(depth)
		mag_list.append(mag)

	for i in range(0, len(Event_list)):
		newx, newy, event=Event_list[i]
		mag, center=event
		lon, lat, depth=center
		cols=pltcolor(depth_list)
		size=pltsize(mag_list)
		if depth_mag:
			ax.scatter(newy-ymin, depth, c=cols[i], s=size[i])
		else:
			ax.scatter(newy-ymin, depth, c='b', s=8)
	return fig