import argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from time import sleep

#command line arguments
parser = argparse.ArgumentParser(description='Plot 3D focal mechanisms')
parser. add_argument('filename', nargs = '?')
parser.add_argument('-r')

args = parser.parse_args()

def parse_file(filename, header):
    #parse tab-delimited file into floats
    data = []
    with open(filename) as f:
        if header: #skip first line
            headers = f.readline()
        for line in f.readlines():
            line = line.strip().split('\t')
            line = [float(x) for x in line]
            x, y, z = line[1:4]
            
            entry = [line[0], [x, y, -1*z], line[4:7]]
            data.append(entry)
    return data

def make_test_data():
    center = [0, 0, 0]
    radius = 1
    data = []
    for strike in range(0, 360, 30):
        for dip in range(-180, 180, 30):
            for slip in range(-180, 180, 30):
                data.append([radius, center, [strike, dip, slip]])

    return data

def scale_beachballs(beachball_list, ax):
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
    for radius, center, angles in beachball_list:
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

def beachball(radius, center, angles, ax, scale_factors, degrees = True):
    '''radius determines the size of the beach ball, center is a list of x,y,z coordinates
    for the center of the beach ball, angles is a list of the strike, dip, and slip angles,
    scale_factors is a list of the proportions to scale the x, y, and z coordinates by to compensate
    for differences in the axes, and degrees is a flag that should be set to True if the strike, dip,
    and slip angles are in degrees and False if they are in radians.'''
    
    points = 30
    colors = ['red', 'white', 'red', 'white']
    borders = [0, np.pi / 2, np.pi, np.pi * 3 / 2]


    v = np.linspace(0, np.pi, points)

    for color, border in zip(colors, borders):
        #generate points for quarter-sphere
        u = np.linspace(border, border + np.pi/2, points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        #combine into coordinate matrix so rotation can be applied
        coordinate_matrix = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        
        slip_rotation = Rotation.from_euler('x', -angles[2], degrees = degrees)
        dip_rotation = Rotation.from_euler('y', 90 - angles[1], degrees = degrees)
        strike_rotation = Rotation.from_euler('z', -angles[0], degrees = degrees)
        slip_rotated = slip_rotation.apply(coordinate_matrix)
        dip_rotated = dip_rotation.apply(slip_rotated)
        strike_rotated = strike_rotation.apply(dip_rotated)

        x = strike_rotated[:, 0]
        y = strike_rotated[:, 1]
        z = strike_rotated[:, 2]

        x = x.reshape(points, points)
        y = y.reshape(points, points)
        z = z.reshape(points, points)

        x = x * radius * scale_factors[0] + center[0]
        y = y * radius * scale_factors[1] + center[1]
        z = z * radius * scale_factors[2] + center[2]
        
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=0)
    

def plot_focal_mechanisms(beachball_list, ax = None, degrees = True):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

    scale_factors = scale_beachballs(beachball_list, ax)
    print(beachball_list)
    for radius, center, angles in beachball_list:
        beachball(radius, center, angles, ax, scale_factors, degrees = degrees)
