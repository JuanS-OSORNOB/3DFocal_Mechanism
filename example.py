#!/usr/bin/env python3
# Last revised: 08/06/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>
import argparse
import numpy as np
import pandas as pd
import os, sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import radians, sin, cos, isclose, asin, atan2
from focal_mechanism import vectors, vec_to_angles, plot_focal_mechanisms

parser = argparse.ArgumentParser(description='Plot 3D focal mechanisms')
parser. add_argument('filename', nargs = '?')
parser.add_argument('-r')

args = parser.parse_args()
#--------------------------------------------------------------------------------------------------------------------
''' EXAMPLE: Importing an Excel FMS dataframe. Creating your beachball list. 
And obtaining the axes' bearing and plunge values in order to plot them on a stereonet.'''

def readingpath(path, file):
	file=os.path.join(path, file)
	if not os.path.isfile(file):
		sys.exit('File(s) missing:'+file)
	return file

workingdir='.'
path=workingdir
focalmechanismdir=readingpath(path, 'FMS.xlsx')

data_FM=pd.read_excel(focalmechanismdir, sheet_name='FMS')
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
for i in range(0,len(mag_FM)):
	beachball_list.append([mag_FM[i], center_FM[i], nodal_plane1[i]])
print('Total number of events:', len(beachball_list))

def plot_test(test_data, lon, lat, depth):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.set_xlabel('Longitude (°)')
	ax.set_ylabel('Latitude (°)')
	ax.set_zlabel('Depth (km)')
	ax.set_xlim(min(lon), max(lon))
	ax.set_ylim(min(lat), max(lat))
	ax.set_zlim(min(depth), max(depth))
	plot_focal_mechanisms(test_data, ax = ax, points = 20,
						  vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'P', 'T']
						  , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
						  print_vecs = True, bottom_half=False)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.view_init(90,270)
	ax.set_xlabel('Longitude (°)')
	ax.set_ylabel('Latitude (°)')
	ax.set_zlabel('Depth (km)')
	ax.set_xlim(min(lon), max(lon))
	ax.set_ylim(min(lat), max(lat))
	ax.set_zlim(min(depth), max(depth))
	plot_focal_mechanisms(test_data, ax = ax, points = 20,
						  vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'P', 'T']
						  , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
						  bottom_half = True)

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

if args.filename == None:
	test_data = beachball_list
	plot_test(test_data, lon, lat, depth)
	plt.show()
	plt.close('all')
else:
	data = parse_file
	plot_focal_mechanisms(parse_file(args.filename, args.r))
	plt.show()
	plt.close('all')

