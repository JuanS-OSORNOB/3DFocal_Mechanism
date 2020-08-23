import inspect, os

import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from focmech3d.datautils import readingfile
from focmech3d.topoprofile import profile_view, in_bounds, plot_profile, plot_bounding_box, pltcolor, pltsize
from focmech3d.focal_mechanism import plot_focal_mechanisms

directory = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

def example(depth_mag=True, verbose = True, show_plots = False):
	'''Example function. It plots the 3D FMS and events inside the bounding box of the profile chosen.
	depth_mag parameter default as True. Please refer to plot_profile function for more details.'''

	#CREATING BEACHBALL LIST
	fm_file=readingfile(os.path.join(directory, 'FMS.xlsx'))
	
	data_FM=pd.read_excel(fm_file, sheet_name='FMS')
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
	if verbose:
		print('Total number of FMS:', len(beachball_list))

	#DEFINING PROFILE PARAMETERS AND CREATING 3D FIGURE
	A=(-74.5, 11.8)
	Aprime=(-72.5, 9.5)
	width=1
	depth=250
	fig=plt.figure(dpi=220)
	ax=fig.add_subplot(111, projection = '3d')
	ax.set_xlabel('Longitude (°)')
	ax.set_ylabel('Latitude (°)')
	ax.set_zlabel('Depth (km)')
	corners, bounds, theta, center, norm_vec = profile_view(*A, *Aprime, width, depth)
	x_A, y_A, x_Aprime, y_Aprime=plot_bounding_box(ax, A, Aprime, corners, depth)
	in_bounds_list=in_bounds(beachball_list, bounds, center, theta, rotated = False)
	plot_focal_mechanisms(in_bounds_list, ax, alpha = 1)
	#plot_focal_mechanisms(beachball_list, ax, alpha=0.5)

	#Plot the events and the focal mechanisms inside profile volume
	events_dir=readingfile('Events.xlsx')
	data_events=pd.read_excel(events_dir)
	df_events=pd.DataFrame(data_events, columns=['latitude', 'longitude', 'depth', 'mag'])

	lon=df_events['longitude'].values.tolist()
	lat=df_events['latitude'].values.tolist()
	depth_events=df_events['depth'].values.tolist()
	mag_events=df_events['mag'].values.tolist()
	#cols_events=pltcolor(depth_events)
	#size_events=pltsize(mag_events)
	#ax.scatter(lon, lat, depth_events, c=cols_events, marker='.', alpha=0.01, edgecolor=cols_events, s=size_events, zorder=-1)	
	
	center_Events=[]
	for i in range(0, len(lon)):
		center_Events.append([lon[i], lat[i], depth_events[i]])
	Event_list=[]
	for i in range(0, len(lon)):
		Event_list.append([mag_events[i], center_Events[i]])
	Events_in_bounds=in_bounds(Event_list, bounds, center, theta, rotated=False)
	
	x_inbound, y_inbound, z_inbound, mag_inbound=[], [], [], []
	for i in Events_in_bounds:
		mag=i[0]
		x, y, z=i[1]
		x_inbound.append(x)
		y_inbound.append(y)
		z_inbound.append(z)
		mag_inbound.append(mag)
	col_inbounds=pltcolor(z_inbound)
	size_inbounds=pltsize(mag_inbound)

	ax.view_init(0, -theta*180/np.pi)
	#ax.view_init(90, 270)

	#PLOTTING PROFILE TOO
	if depth_mag:
		ax.scatter(x_inbound, y_inbound, z_inbound, c=col_inbounds, marker='.', alpha=0.05, edgecolor=col_inbounds, s=size_inbounds, zorder=-1)
		fig2 = plot_profile(in_bounds_list, Event_list, x_A, y_A, x_Aprime, y_Aprime, width, depth, depth_mag=True, 
		Title='Profile plot', verbose = verbose)
	else:
		ax.scatter(x_inbound, y_inbound, z_inbound, c='b', marker='.', alpha=0.05, edgecolor='b', s=10, zorder=-1)
		fig2 = plot_profile(in_bounds_list, Event_list, x_A, y_A, x_Aprime, y_Aprime, width, depth, depth_mag=False, 
		Title='Profile plot', verbose = verbose)
	if show_plots:
		plt.show()
		plt.close('all')
	return fig, fig2

if __name__ == '__main__':
	example(depth_mag=True, verbose = True, show_plots = True)


