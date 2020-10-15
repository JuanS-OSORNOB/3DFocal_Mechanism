#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import os, sys, inspect
from math import radians, sin, cos, isclose, asin, atan2, degrees

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.testing.compare import compare_images

from focmech3d.focal_mechanism import plot_focal_mechanisms
from focmech3d.vector_math import vectors, vec_to_angles
from focmech3d.datautils import readingfile
from focmech3d import load_fms

#--------------------------------------------------------------------------------------------------------------------
''' EXAMPLE: Importing an Excel FMS dataframe. Creating your beachball list. 
And obtaining the axes' bearing and plunge values in order to plot them on a stereonet.'''

#This is just to make sure the script can find the companion file used in the example.
directory = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
focalmechanismdir=readingfile(os.path.join(directory, 'FMS.xlsx'))
def filepath(filename):
	return os.path.join(directory, filename)
def savefig(fig, filename):
	fig.savefig(filepath(filename))


data_FM=load_fms(focalmechanismdir, filetype = 'excel', usecols = [4, 3, 13, 12, 5, 6, 7, 8, 9, 10, 14], invert_z = True)
print('Total number of events:', len(data_FM))



def plot_test(test_data, view_init = [], filename = '', bottom_half = True):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.set_xlabel('Longitude (°)')
	ax.set_ylabel('Latitude (°)')
	ax.set_zlabel('Depth (km)')

	plot_focal_mechanisms(test_data, ax = ax, points = 20,
						  vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'P', 'T']
						  , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
						  bottom_half = bottom_half, in_fms = True)
	if view_init:
		ax.view_init(*view_init)
	if filename:
		savefig(fig, filename)

'''If you would like to use the second nodal plane it is also possible. 
Whichever you use can yield the correct 3D and stereonet plots, as nodal planes are perpendicular.'''

nodal_plane1 = [[degrees(fm.strike), degrees(fm.dip), degrees(fm.rake)] for fm in data_FM]
nodal_plane2 = [(fm.other_params_dict['Strike 2'], fm.other_params_dict['Dip 2'], fm.other_params_dict['Rake 2']) for fm in data_FM]

test = data_FM[0]
print(test.other_params_dict['Strike 2'])
print(test.other_params_dict['Dip 2'])
print(test.other_params_dict['Rake 2'])
print(test.aux_plane_angles())

def obtain_axes_list(plane):
	bearing=[]
	plunge=[]
	index=0
	for i in plane:
		index+=1
		vecs=vectors(i)
		#print('Earthquake #',index)
		for v in vecs:
			bearing1, plunge1=vec_to_angles(vecs[v])
			#print(bearing,plunge)
			bearing.append(bearing1)
			plunge.append(plunge1)
	Strike=bearing[0::7]
	Dip=plunge[1::7]
	
	b_bearing=bearing[4::7]
	b_plunge=plunge[4::7]
	
	p_bearing=bearing[5::7]
	p_plunge=plunge[5::7]
	
	t_bearing=bearing[6::7]
	t_plunge=plunge[6::7]
	#print(Strike,len(Strike))
	#print(Dip, len(Dip))
	#print(p_bearing, len(p_bearing))
	return b_bearing, b_plunge, p_bearing, p_plunge, t_bearing, t_plunge

b_bearing1, b_plunge1, p_bearing1, p_plunge1, t_bearing1, t_plunge1=obtain_axes_list(nodal_plane1)
b_bearing2, b_plunge2, p_bearing2, p_plunge2, t_bearing2, t_plunge2=obtain_axes_list(nodal_plane2)
#With these lists you can use the mplstereonet module to plot these axes on the stereonet

'''Note: Area query in case it needs to be done'''
AREA1 = []
AREA2 = []

for fm in data_FM:
	area = fm.other_params_dict['Area']
	if area == 1:
		AREA1.append(fm)
	elif area == 2:
		AREA2.append(fm)


plot_test(data_FM, filename = 'example_plot1.png', bottom_half = False)
plot_test(data_FM, view_init = (90, 270), filename = 'example_plot2.png')
# plt.show()
# plt.close('all')

print(compare_images(filepath('old_example_plot1.png'), filepath('new_example_plot1.png'), .01))
print(compare_images(filepath('old_example_plot2.png'), filepath('new_example_plot2.png'), .01))
