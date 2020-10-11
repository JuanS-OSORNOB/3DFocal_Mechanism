#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import os, sys, inspect
from math import radians, sin, cos, isclose, asin, atan2

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
	fig.savefig(os.path.join(directory, filename))

data_FM = pd.read_excel(focalmechanismdir)

df_FM=pd.DataFrame(data_FM, columns=['Longitude (°)', 'Latitude (°)', 'Depth (km)', 'Magnitude (Mw)', 'Strike 1', 'Dip 1', 'Rake 1', 'Strike 2', 'Dip 2', 'Rake 2', 'Area', 'Date'])

mag_FM, lon, lat, depth, center_FM, nodal_plane1=[], [], [], [], [], []
for i, row in df_FM.iterrows():
		mag_FM.append(row['Magnitude (Mw)'])
		x_FM=row['Longitude (°)']
		y_FM=row['Latitude (°)']
		z_FM=row['Depth (km)']*(-1)
		s_FM=row['Strike 1']
		d_FM=row['Dip 1']
		r_FM=row['Rake 1']
		lon.append(x_FM)
		lat.append(y_FM)
		depth.append(z_FM)
		center_FM.append([x_FM, y_FM, z_FM])
		nodal_plane1.append([s_FM, d_FM, r_FM])
#Insert beachball list=[[R1, [X1,Y1,Z1], [S1, D1, R1]],[[R2, [X2,Y2,Z2], [S2, D2, R2]],...,[[Ri, [Xi,Yi,Zi], [Si, Di, Ri]]] from i=1 to n number of FMS.
beachball_list=[]
data_FM=load_fms(focalmechanismdir, filetype = 'excel', usecols = [4, 3, 13, 12, 5, 6, 7])
for i in range(0,len(mag_FM)):
	beachball_list.append([mag_FM[i], center_FM[i], nodal_plane1[i]])
print('Total number of events:', len(data_FM))

def plot_test(test_data, lon, lat, depth):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.set_xlabel('Longitude (°)')
	ax.set_ylabel('Latitude (°)')
	ax.set_zlabel('Depth (km)')

	plot_focal_mechanisms(test_data, ax = ax, points = 20,
						  vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'P', 'T']
						  , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
						  print_vecs = True, bottom_half=False)
	fig.savefig('new_example_plot1.png')

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.view_init(90,270)
	ax.set_xlabel('Longitude (°)')
	ax.set_ylabel('Latitude (°)')
	ax.set_zlabel('Depth (km)')

	plot_focal_mechanisms(test_data, ax = ax, points = 20,
						  vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'P', 'T']
						  , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
						  bottom_half = True)
	ax.view_init(90,270)
	fig.savefig('new_example_plot2.png')
'''If you would like to use the second nodal plane it is also possible. 
Whichever you use can yield the correct 3D and stereonet plots, as nodal planes are perpendicular.'''
strike2=df_FM['Strike 2'].values.tolist()
dip2=df_FM['Dip 2'].values.tolist()
rake2=df_FM['Rake 2'].values.tolist()
nodal_plane2=[]
for i in range(0, len(strike2)):
	nodal_plane2.append((strike2[i], dip2[i], rake2[i]))

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
AREA1_FM=df_FM[df_FM['Area'].eq(1)]
AREA2_FM=df_FM[df_FM['Area'].eq(2)]

test_data = beachball_list
plot_test(test_data, lon, lat, depth)
# plt.show()
# plt.close('all')

print(compare_images('old_example_plot1.png', 'new_example_plot1.png', .01))
print(compare_images('old_example_plot2.png', 'new_example_plot2.png', .01))
