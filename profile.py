from focal_mechanism import plot_focal_mechanisms, vectors, scale_beachballs, plot_vector
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import isclose, atan2, cos, sin, sqrt
import os, sys
import pandas as pd
from matplotlib.testing.compare import compare_images
def readingfile(path, filename):
	file=os.path.join(path, filename)
	if not os.path.isfile(file):
		sys.exit('File(s) missing:'+file)
	return path, file

def createpath(directory):
	if not os.path.isdir(directory):
		os.mkdir(directory)
	return directory

def translate_rotate_point(x, y, angle, center):
	'''Move a point (x, y) so that the new point is the same with respect
	to the origin as the point (x, y) was with respect to the point 'center'.
	Then rotate the new point around the origin by 'angle', counterclockwise from the
	x-axis.'''
	x, y = x - center[0], y - center[1]
	newx = x * cos(angle) - y * sin(angle)
	newy = x * sin(angle) + y * cos(angle)
	return newx, newy

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

def normalize_vector(vector):
	#cast to float
	vector = vector.astype(float)
	return vector / np.linalg.norm(vector)

def angle_between(vec1, vec2):
	vec1 = normalize_vector(vec1)
	vec2 = normalize_vector(vec2)
	dotprod = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
	return np.arccos(dotprod)

def circle_angle(axis1, axis2, vec):
	'''Returns the angle(in radians) between vec and axis1. Uses axis2 for directionality
	(i.e. axis2 is at pi/2 rather than 3pi/2).'''
	angle = angle_between(axis1, vec)
	chirality = angle_between(axis2, vec)
	if chirality > np.pi/2:
		return 2 * np.pi - angle
	return angle

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

def profile_bounding_box(x1, y1, x2, y2, width):
	'''This function characterizes the rectangle that is the upper face of the profile (see profile_view for more details about profiles).

	This function returns the xy coordinates of the four corners and center of the rectangle, the vector pointing directly toward the viewer
	(normal to the viewing plane), and the amount of rotation of the rectangle midline clockwise 
	from the y-axis, in radians. The rotation angle doubles as the azimuthal viewing angle to be given to Matplotlib so that if a 3D 
	plot of the profile is made, the viewer will be looking perpendicular to the line between (x1, y1) and (x2, y2).'''
	
	pass

def bounding_box_corners():
	pass

def profile_view(x1, y1, x2, y2, width, depth):
	'''A profile is a view of a slice of terrain and associated events. It consists of a right rectangular prism whose upper face is 
	on the surface (at depth 0). The surface rectangle is defined by the variables x1, y1, x2, y2, and width. (x1, y1) and (x2, y2) are 
	points on the surface (in the XY plane). The line segment from (x1, y1) to (x2, y2) is the midline of the surface rectangle, which therefore
	has two parallel sides whose midpoints are (x1, y1) and (x2, y2) and whose length is 'width' units. The lower face of the prism is
	'depth' units below the surface and has the same xy coordinates as the upper face. The view is perpendicular to this midline, so that 
	the viewing plane has the left side corresponding to (x1, y1), the right side corresponding to (x2, y2), the bottom corresponding to (-depth) 
	and the top corresponding to the surface. 
	
	[Describe what this function actually does]'''



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
	

def scale_factors_2d(ax):
	ratio = ax.get_data_ratio()
	if ratio < 1:
		return (1, ratio)
	return (1 / ratio, 1)

def plot_profile(data_list, x1, y1, x2, y2, width, depth):
	original_corners, bounds, theta, center, norm_vec = profile_view(x1, y1, x2, y2, width, depth)
	in_bounds_list = in_bounds(data_list, bounds, center, theta)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	_, _, ymin, ymax, zmin, zmax = bounds
	ax.set_xlim(0, ymax - ymin)
	ax.set_ylim(zmin, zmax)
	in_bounds_list.sort() #first value was just for sorting back to front
	scale_factors = scale_factors_2d(ax)
	for i in range(len(in_bounds_list)):
		_, x, event = in_bounds_list[i]
		radius, center, angles = event
		vecs = vectors(angles)
		plot_lambert(ax, (x - ymin, center[2]), radius, scale_factors, i, norm_vec, np.array([0, 0, 1]), vecs)
	return fig, ax

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

def pointprofile(fig, ax, file, carpeta, sheets, columns, fignames, **kwargs):
	print('POINT PROFILES')
	for i, sheet in enumerate(sheets):
		data=pd.read_excel(file, sheet_name=sheet)
		df=pd.DataFrame(data, columns=columns)
		m=df[columns[0]].values.tolist()
		PP=df[columns[1]].values.tolist()
		mag=df[columns[2]].values.tolist()
		min_m=min(m)
		max_m=max(m)
		min_PP=min(PP)
		max_PP=max(PP)
		fig=plt.figure(dpi=220, tight_layout=False)
		ax=fig.add_subplot(111)
		figname=fignames[i]
		ax.set_title('{}'.format(figname))
		ax.set_aspect('equal')
		if 'xlabel' in kwargs:
			ax.set_xlabel(kwargs['xlabel'], fontsize=8)
		if 'ylabel' in kwargs:	
			ax.set_ylabel(kwargs['ylabel'], fontsize=8)
		cols=pltcolor(PP)
		size=pltsize(mag)
		plt.scatter(m, PP, c=cols, marker='.', s=size)
		plt.grid(which='major', axis='x', linestyle='--', alpha=0.5)
		plt.show()
		os.chdir(carpeta)
		fig.savefig('{}'.format(sheet))
		plt.close('all')
	print('Find figures at', carpeta)


	
	

#EXAMPLE


def main():
	data = [[1, [0, 0, 0], [20, 3, 45]],
			[3, [3, 5, -2], [40, 10, 60]],
			[5, [2, 15, -10], [0, 34, 40]],
			[5, [2.2, 16, -10], [0, 90, 40]],
			[3, [1.5, 11, -20], [10, 40, 20]]]

	
	A = (0, 10)
	Aprime = (3, 30)
	width = 4
	depth = 100
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	corners, bounds, theta, center, norm_vec = profile_view(*A, *Aprime, width, depth)
	in_bounds_list = in_bounds(data, bounds, center, theta, rotated = False)
	norm_vec = np.array([norm_vec[0], norm_vec[1], 0])

	x_A, y_A = A
	x_Aprime, y_Aprime = Aprime
	
	ax.scatter(x_A, y_A, c='k')
	ax.scatter(x_Aprime, y_Aprime, c='k')
	ax.text(x_A, y_A, 0, 'A', size=15)
	ax.text(x_Aprime, y_Aprime, 0, "A'", size=15)

	#plot corner points -- corners have already been defined by the profile_view function
	corner_labels = ['V1', 'V2', 'V4', 'V3']
	for v, label in zip(corners, corner_labels):
		ax.scatter(*v, c = 'k')
		ax.text(*v, label, size = 10)

	#plot line from A to Aprime
	x_values=[x_A, x_Aprime]
	y_values=[y_A, y_Aprime]
	ax.plot(x_values, y_values, c='purple')
	
	#plot bounding box

	#create lower corner coordinates by subtracting the depth from the z value
	lower_corners = [v + np.array([0, 0, -depth]) for v in corners]

	#plot vertical lines between pairs of upper and lower corners
	for V_upper, V_lower in zip(corners, lower_corners):
		ax.plot(*zip(V_upper, V_lower), c = 'k', linestyle = 'dotted')

	#add first value of upper and lower rectangle to list so that the line forms a complete box
	five_corners = list(corners) + [corners[0]]
	five_lower_corners = lower_corners + [lower_corners[0]]

	#plot upper and lower rectangles
	ax.plot(*zip(*five_corners), c='k', linestyle='dashed')
	ax.plot(*zip(*five_lower_corners), c='k', linestyle='dashed')

	#plot the focal mechanisms inside the profile volume
	plot_focal_mechanisms(in_bounds_list, ax, alpha = 1)
	ax.view_init(0, -theta*180/np.pi)

	#also plot the profile
	fig2, ax2 = plot_profile(in_bounds_list, x_A, y_A, x_Aprime, y_Aprime, width, depth)
	plt.show()
	path1, file1=readingfile('/home/juan/Caribe_2020/Point Profiles/', 'Sismos_Perfiles_Tabla.xls')
	directory1=os.path.join(path1, 'Figures')
	carpeta1=createpath(directory1)
	sheets1=['Sismos_PerfilAA_Tabla', 'Sismos_PerfilBB_Tabla', 'Sismos_PerfilCC_Tabla', 'Sismos_PerfilDD_Tabla', 'Sismos_PerfilEE_Tabla']
	columns1=['x_km', 'pp', 'mag']
	fignames1=['AA´', 'BB´', 'CC´', 'DD´', 'EE´']
	pointprofile(fig2, ax2, file1, carpeta1, sheets1, columns1, fignames1, xlabel='label x', ylabel='label y')


if __name__ == '__main__':
	main()