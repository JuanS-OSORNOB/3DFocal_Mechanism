import argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from math import radians, sin, cos, isclose, asin, atan2

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

def plot_circle(radius, center, vecs, ax, scale_factors, fault_color = 'black', auxiliary_color = 'blue',
                degrees = True):
    strike = vecs['strike']
    dip = vecs['dip']
    normal = vecs['normal']
    null = vecs['null']
    u = np.linspace(0, 2*np.pi)

    #fault plane, defined by strike and dip vectors which are orthogonal and both in the plane
    x = strike[0] * np.cos(u) + dip[0] * np.sin(u)
    y = strike[1] * np.cos(u) + dip[1] * np.sin(u)
    z = strike[2] * np.cos(u) + dip[2] * np.sin(u)

    x = x * radius * scale_factors[0] + center[0]
    y = y * radius * scale_factors[1] + center[1]
    z = z * radius * scale_factors[2] + center[2]

    ax.plot(x, y, z, color = fault_color, linewidth = 2)
    
    #auxiliary plane, defined by normal and null vectors which are orthogonal and both in the plane
    x = normal[0] * np.cos(u) + null[0] * np.sin(u)
    y = normal[1] * np.cos(u) + null[1] * np.sin(u)
    z = normal[2] * np.cos(u) + null[2] * np.sin(u)

    x = x * radius * scale_factors[0] + center[0]
    y = y * radius * scale_factors[1] + center[1]
    z = z * radius * scale_factors[2] + center[2]



    ax.plot(x, y, z, color = auxiliary_color, linewidth = 2)

def plot_vector(radius, center, vec, ax, scale_factors, color):
    v = vec * scale_factors
    ax.quiver(*center, *v, colors = color, length = radius)

def vectors(angles, degrees = True):
    if degrees:
        strike, dip, rake = [radians(x) for x in angles]
    else:
        strike, dip, rake = angles

    strike_vector = np.array([sin(strike),
                              cos(strike),
                              0])
    dip_vector = np.array([cos(dip)*cos(strike),
                           -cos(dip)*sin(strike),
                           -sin(dip)])
    rake_vector = np.array([cos(rake)*sin(strike) - sin(rake)*cos(dip)*cos(strike),
                            cos(rake)*cos(strike) + sin(rake)*cos(dip)*sin(strike),
                            sin(rake)*sin(dip)])
    normal_vector = np.array([sin(dip)*cos(strike),
                              -sin(dip)*sin(strike),
                              cos(dip)])
    null_vector = np.array([-sin(rake)*sin(strike) - cos(rake)*cos(dip)*cos(strike),
                            -sin(rake)*cos(strike) + cos(rake)*cos(dip)*sin(strike),
                            cos(rake)*sin(dip)])

    p_vector = np.array([sin(dip)*cos(strike) - cos(rake)*sin(strike) + sin(rake)*cos(dip)*cos(strike),
                         -sin(dip)*sin(strike) - cos(rake)*cos(strike) - sin(rake)*cos(dip)*sin(strike),
                         cos(dip) - sin(rake)*sin(dip)])

    t_vector = np.array([sin(dip)*cos(strike) + cos(rake)*sin(strike) - sin(rake)*cos(dip)*cos(strike),
                         -sin(dip)*sin(strike) + cos(rake)*cos(strike) + sin(rake)*cos(dip)*sin(strike),
                         cos(dip) + sin(rake)*sin(dip)])
    #sanity checks

    #normal vector should be the cross product of dip vector and strike vector
    norm_correct = np.isclose(normal_vector, np.cross(dip_vector, strike_vector))   
    assert(norm_correct.all())
    
    #rake vector should be cos(rake) * strike_vector - sin(rake) * dip_vector
    rake_correct = np.isclose(rake_vector, cos(rake) * strike_vector - sin(rake) * dip_vector)                            
    assert(rake_correct.all())
    
    #null vector should be the cross product of normal vector and rake vector
    null_correct = np.isclose(null_vector, np.cross(normal_vector, rake_vector))
    assert(null_correct.all())

    #p should be normal - rake
    p_correct = np.isclose(p_vector, normal_vector - rake_vector)
    assert(p_correct.all())
    
    #t should be normal + rake
    t_correct = np.isclose(t_vector, normal_vector + rake_vector)
    assert(t_correct.all())

    #normalize p and t so they have a length of 1
    p_vector = p_vector / np.linalg.norm(p_vector)
    t_vector = t_vector / np.linalg.norm(t_vector)

    #check if null, p, and t are pointing downward. If not, reverse them.
    if null_vector[2] > 0:
        null_vector = null_vector * -1
    if p_vector[2] > 0:
        p_vector = p_vector * -1
    if t_vector[2] > 0:
        t_vector = t_vector * -1

    return {'strike': strike_vector,
            'dip' : dip_vector,
            'rake' : rake_vector,
            'normal' : normal_vector,
            'null': null_vector,
            'p': p_vector,
            't': t_vector}

def vec_to_angles(vector):
    '''takes an xyz vector and returns bearing (degrees clockwise from y axis) and
    plunge (degrees below horizontal plane) angles.'''
    
    x, y, z = vector
    mag = np.linalg.norm(vector)
    bearing = atan2(x, y) * 180/np.pi
    plunge = -asin(z/mag) * 180/np.pi

    return bearing, plunge

def print_vectors(vecs):
    '''Takes a dict of xyz vectors, prints the vector type, xyz vector, and plunge/bearing format.'''

    textstring = '{0}: <{1},{2},{3}>, bearing: {4} degrees, plunge: {5} degrees'

    for v in vecs:
        bearing, plunge = vec_to_angles(vecs[v])
        #shorten to two decimal places
        shortened = ['{:.2f}'.format(x) for x in [*vecs[v], bearing, plunge]]
        print(textstring.format(v, *shortened))
        
        
        
        
        
    
    

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


def plot_focal_mechanisms(data_list, ax = None, **kwargs):
    '''kwargs:
            degrees: True or False (default True).
                If True, strike, dip, and rake angles are given
                in degrees. If False, they are given in radians.
            bottom_half: True or False (default False).
                When True, shows the bottom
                half of the sphere only, with the top half removed.
            alpha: Float between 0 and 1 (default .75).
                Sets transparancy of the focal
                mechanism sphere, with 0 being fully transparent and 1 being fully opaque
            points: Integer (default 20).
                The number of points used to make the sphere. Higher
                numbers have better resolution but make take longer to render.
            plot_planes: True or False (default True).
                If True, plot great circles that represents the focal planes.
            vector_plots: List of vectors to plot.
            vector_colors: List of colors for each vector.'''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

    scale_factors = scale_beachballs(data_list, ax)
    for radius, center, angles in data_list:
        focal_mechanism(radius, center, angles, ax, scale_factors, **kwargs)
    if 'vector_plots' in kwargs:
        #make proxy legend
        for label, color in zip(kwargs['vector_plots'], kwargs['vector_colors']):
            ax.plot([], [], label = label, color = color)
    plt.legend()

def focal_mechanism(radius, center, angles, ax, scale_factors, degrees = True, bottom_half = False,
                    alpha = .75, points = 20, plot_planes = True, vector_plots = [], vector_colors = [],
                    print_vecs = False):
    '''radius determines the size of the beach ball, center is a list of x,y,z coordinates
    for the center of the beach ball, angles is a list of the strike, dip, and slip angles,
    scale_factors is a list of the proportions to scale the x, y, and z coordinates by to compensate
    for differences in the axes, and degrees is a flag that should be set to True if the strike, dip,
    and slip angles are in degrees and False if they are in radians.'''
    
    colors = ['red', 'white', 'red', 'white']
    borders = [0, np.pi / 2, np.pi, np.pi * 3 / 2]


    v = np.linspace(0, np.pi, points)

    vecs = vectors(angles)
    p = vecs['p']
    t = vecs['t']

    for color, border in zip(colors, borders):
        #generate points for quarter-sphere
        u = np.linspace(border, border + np.pi/2, points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        #combine into coordinate matrix so rotation can be applied
        coordinate_matrix = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        #apply rotations to matrix
        slip_rotation = Rotation.from_euler('x', angles[2], degrees = degrees)
        dip_rotation = Rotation.from_euler('y', angles[1] - 90, degrees = degrees)
        strike_rotation = Rotation.from_euler('z', -angles[0], degrees = degrees)
        slip_rotated = slip_rotation.apply(coordinate_matrix)
        dip_rotated = dip_rotation.apply(slip_rotated)
        strike_rotated = strike_rotation.apply(dip_rotated)

        #separate x, y, and z matrices
        x = strike_rotated[:, 0]
        y = strike_rotated[:, 1]
        z = strike_rotated[:, 2]

        #unflatten
        x = x.reshape(points, points)
        y = y.reshape(points, points)
        z = z.reshape(points, points)

        #remove the top half of the sphere
        if bottom_half:
            #for each point, determine if the line between the point above it
            #and/or to the left (in the grid, not in xyz space), crosses
            #the xy plane.
            for i in range(points):
                for j in range(points):
                    if i != 0 and z[i, j] * z[i - 1, j] < 0:
                        shorten_line(x, y, z, i, j, i - 1, j)
                    if j != 0 and z[i, j] * z[i, j - 1] < 0:
                        shorten_line(x, y, z, i, j, i, j - 1)
                      
            x[np.where(z > 0)] = np.nan

        #multiply by radius to resize, by scale_factors to compensate for axis size differences, and add center
        #coordinates to translate to the correct location
        x = x * radius * scale_factors[0] + center[0]
        y = y * radius * scale_factors[1] + center[1]
        z = z * radius * scale_factors[2] + center[2]


        
        ax.plot_surface(x, y, z, color=color, linewidth=0, alpha = alpha)

    if plot_planes:
        plot_circle(radius, center, vecs, ax, scale_factors, degrees = degrees)

    for vectype, c in zip(vector_plots, vector_colors):
        vec = vecs[vectype]
        plot_vector(radius, center, vec, ax, scale_factors, c)

    if print_vecs:
        print('Strike: {} degrees, Dip: {} degrees, Rake: {} degrees'.format(*angles))
        print_vectors(vecs)
            

        

def shorten_line(x, y, z, i, j, i2, j2):
    '''shorten line between <x[i,j], y[i,j], z[i,j]>
    and <x[i2,j2], y[i2,j2], z[i2,j2]> so that the point above the xy
    plane lies on it'''
    if z[i, j] < 0:
        #if z[i, j] is the smaller of the two, switch indices so it's larger
        i, j, i2, j2 = i2, j2, i, j
    #now <x[i,j], y[i,j], z[i,j]> is the point to be moved

    #calculate fraction of line to remove in each dimension
    zfrac = z[i, j] / (z[i, j] - z[i2, j2])
    xdist = zfrac * (x[i, j] - x[i2, j2])
    ydist = zfrac * (y[i, j] - y[i2, j2])

    z[i, j] = 0
    x[i, j] = x[i, j] - xdist
    y[i, j] = y[i, j] - ydist
    
def plot_test():

    data = [[1, [2, 0, 0], [0, 20, 45]]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    plot_focal_mechanisms(data, ax = ax, points = 20,
                          vector_plots = ['strike', 'dip', 'rake', 'normal', 'null', 'p', 't']
                          , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
                          print_vecs = True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    plot_focal_mechanisms(data, ax = ax, points = 20,
                          vector_plots = ['strike', 'dip', 'rake', 'normal', 'null', 'p', 't']
                          , vector_colors = ['blue', 'green', 'brown', 'black', 'purple', 'gray', 'red'],
                          bottom_half = True)




if args.filename == None:
    plot_test()
    plt.show()
else:
    data = parse_file
    plot_focal_mechanisms(parse_file(args.filename, args.r))
    plt.show()
        
    
    
