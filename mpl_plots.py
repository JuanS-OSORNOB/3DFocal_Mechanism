import numpy as np
from plotcoords import circle_arc, translate_and_scale

def generate_scale_factors(focalmechanisms, ax):
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
	for fm in focalmechanisms:
		center = fm.location
		radius = fm.magnitude
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

def plot_circle(focalmechanism, ax, axis_ratios, fault_color = 'black', auxiliary_color = 'blue',
				degrees = True):
				
	strike = focalmechanism.vectors['strike']
	dip = focalmechanism.vectors['dip']
	normal = focalmechanism.vectors['normal']
	null = focalmechanism.vectors['B']
	radius = focalmechanism.magnitude
	scale_factors = [radius * x for x in axis_ratios]

	#fault plane, defined by strike and dip vectors which are orthogonal and both in the plane
	coords = circle_arc(strike, dip, 0, 2 * np.pi)
	x, y, z = translate_and_scale(coords, focalmechanism.location, scale_factors)
	ax.plot(x, y, z, color = fault_color, linewidth = 2)
	
	#auxiliary plane, defined by normal and null vectors which are orthogonal and both in the plane
	coords = circle_arc(normal, null, 0, 2 * np.pi)
	x, y, z = translate_and_scale(coords, focalmechanism.location, scale_factors)
	ax.plot(x, y, z, color = auxiliary_color, linewidth = 2)


def plot_vector(radius, center, vec, ax, scale_factors, color):
	v = vec * scale_factors
	ax.quiver(*center, *v, colors = color, length = radius)
