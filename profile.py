from focal_mechanism import plot_focal_mechanisms, vectors, scale_beachballs, plot_vector
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import isclose, atan2, cos, sin

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

    return x, y, z

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
    ap_orth = np.cross(vecs['rake'], ap_int)
    if np.dot(ap_orth, projection_point) < 0:
        ap_orth *= -1
    
    #use null to determine intersection point
    null = vecs['B']
    #make sure null is in near hemisphere
    if np.dot(null, projection_point) < 0: 
        null *= -1

    #use t-vector to determine which quadrants to fill
    tension = vecs['T']
    #make sure T is in near-hemisphere
    if np.dot(tension, projection_point) < 0:
        tension *= -1
        
    fp_intersect_angle = circle_angle(fp_int, fp_orth, null)
    ap_intersect_angle = circle_angle(ap_int, ap_orth, null)

    fp_int_null_arc = np.array(circle_arc(fp_int, fp_orth, 0, fp_intersect_angle))
    null_neg_fp_int_null_arc = np.array(circle_arc(fp_int, fp_orth, fp_intersect_angle, np.pi))

    ap_int_null_arc = np.array(circle_arc(ap_int, ap_orth, 0, ap_intersect_angle))
    null_neg_ap_int_arc = np.array(circle_arc(ap_int, ap_orth, ap_intersect_angle, np.pi))

    angles = [circle_angle(new_y_axis, new_x_axis, vec) for vec in [fp_int, ap_int, -fp_int, -ap_int]]
    fp_angle, ap_angle, neg_fp_angle, neg_ap_angle = angles
    
    #case1: ap_int is clockwise from fp_int (assuming new_y_axis at 12 and new_x_axis at 3)
    #case2: ap_int is counterclockwise from fp_int

    if fp_angle > ap_angle and fp_angle - ap_angle < np.pi:
        '''case 2: shapes are:
                ap -> fp -> intersection -> ap
                fp -> -ap -> intersection -> fp
                -ap -> -fp -> intersection -> -ap
                -fp -> ap -> intersection -> -fp'''
        arc = np.array(circle_arc(new_y_axis, new_x_axis, ap_angle, fp_angle))
        shape1 = np.concatenate(arc, fp_int_null_arc, null_ap_int_arc)
        shape2 = circle_arc(new_y_axis, new_x_axis, fp_angle, neg_ap_angle) + null_neg_ap_int_arc[::-1] + fp_int_null_arc[::-1]
    
                
        
    else:
        '''case1: shapes are:
                fp -> ap -> intersection -> fp
                ap -> -fp -> intersection -> ap
                -fp -> -ap -> intersection -> -fp
                -ap -> fp -> intersection -> -ap'''
        arc = np.array(circle_arc(new_y_axis, new_x_axis, fp_angle, ap_angle))
        shape1 = np.concatenate((arc, ap_int_null_arc, fp_int_null_arc[:, ::-1]), axis = 1)
        
    #shape1: from fp_int to intersection, from intersection to ap_int, from ap_int to fp_int
##    fp_angle = angles[0]
##    ap_angle = angles[1]
##    arc1 = fp_int_null_arc
##    arc2 = _ap_int_arc
##    if fp_angle < ap_angle:
##        arc3 =  circle_arc(new_y_axis, new_x_axis, fp_angle, ap_angle)
##        arc1 = arc1[::-1]
##        arc2 = arc2[::-1]
##        arc3 = arc3[::-1]
##    else:
##        arc3 = circle_arc(new_y_axis, new_x_axis, ap_angle, fp_angle)
        
        
        
    #shape2: from -fp_int to intersection, from intersection to ap_int, from ap_int to -fp_int
    #shape3: from fp_int to intersection, from intersection to -ap_int, from -ap_int to fp_int
    #shape4: from -fp_int to intersection, from intersection to -ap_int, from -ap_int to -fp_int

    
    
    arc_set = []
    #outer circle
    if angle_between(fp_int, ap_int) < np.pi:
        angles = [circle_angle(new_y_axis, new_x_axis, vec) for vec in [fp_int, ap_int, -fp_int, -ap_int]]
    else:
        angles = [circle_angle(new_y_axis, vec) for vec in [fp_int, -ap_int, -fp_int, ap_int]]
    for i in range(-1, len(angles) - 1):
        arc_set.append(circle_arc(new_y_axis, new_x_axis, angles[i], angles[i + 1]))

    arc_set.append(circle_arc(fp_int, fp_orth, 0, fp_intersect_angle))
    arc_set.append(circle_arc(fp_int, fp_orth, fp_intersect_angle, np.pi))
    arc_set.append(circle_arc(ap_int, ap_orth, 0, ap_intersect_angle))
    arc_set.append(circle_arc(ap_int, ap_orth, ap_intersect_angle, np.pi))

    arc_set = [np.array(x) for x in arc_set]
    arc_set.append(shape1)

    coords = []
    for arc in arc_set:
        coords.append(xyz_to_lambert(arc[0, :], arc[1, :], arc[2, :], projection_point, new_x_axis, new_y_axis))



    return coords
    

def profile_view(data_list, x1, y1, x2, y2, width, depth):
    '''Quick and dirty function for looking at a slice from the side. Will
    improve later. (x1, y1) and (x2, y2) are points in the XY plane that
    are the endpoints of a line down the center of the rectangle that is the
    bounding box. Width is the width of that box, and depth is the depth of
    the box.'''



    #vector in direction of midline
    vec1 = np.array([x2 - x1, y2 - y1, 0])
    #viewing plane is vertical so another vector is z unit vector
    vec2 = np.array([0, 0, 1])
    #vector normal to both of these is normal to the plane
    norm_vec = np.cross(vec1, vec2)
    #normal vector should be in XY plane, so
    assert(norm_vec[2] == 0)
    norm_vec = norm_vec / np.linalg.norm(norm_vec)
    norm_vec = norm_vec[:2]
    
    #find limits of bounding box
    corner1 = np.array([x1, y1]) - width/2 * norm_vec
    corner2 = corner1 + width * norm_vec
    corner3 = np.array([x2, y2]) + width/2 * norm_vec
    corner4 = corner3 - width * norm_vec
    original_corners = (corner1, corner2, corner3, corner4)
    
    #find center of bounding box (midpoint of diagonal)
    centerx = (corner1[0] + corner3[0]) / 2
    centery = (corner1[1] + corner3[1]) / 2
    translate = np.array([centerx, centery])

    #find rotation angle (negative because we want to reverse the angle)
    theta = -atan2(norm_vec[1], norm_vec[0])
    
    #recenter and rotate corners
    corners = [translate_rotate_point(*corner, theta, translate) for corner in original_corners]
    corner1, corner2, corner3, corner4 = corners
    #establish bounds
    xmin = corner1[0]
    if isclose(corner2[0], xmin):
        #then corner1 and corner2 are on same vertical line
        #therefore corner3 is on the other vertical line
        xmax = corner3[0]
        #corner4 should be on same vertical line as corner3
        assert(isclose(xmax, corner4[0]))
    else:
        #then corner4 should be on same vertical line as corner1
        assert(isclose(xmin, corner4[0]))
        #and corner2 is on a different vertical line
        xmax = corner2[0]
        #and corner3 should be on the same vertical line as corner2
        assert(isclose(xmax, corner3[0]))
    if xmin > xmax:
        #switch xmin and xmax if xmin is bigger
        xmin, xmax = xmax, xmin
    
    #now do the same thing to find ymin and ymax
    ymin = corner1[1]
    if isclose(corner2[1], ymin):
        ymax = corner3[1]
        assert(isclose(ymax, corner4[1]))
    else:
        assert(isclose(ymin, corner4[1]))
        ymax = corner2[1]
        assert(isclose(ymax, corner3[1]))
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    #the z-value of a focal mechanism is expected to be a negative number, so depth
    #should start at a negative number and go to 0
    #or maybe modify this later to specify both ends of depth range

    if depth > 0:
        depth *= -1
    zmin = depth
    zmax = 0
    in_bounds = []

    for event in data_list:
        x, y, z = event[1]
        newx, newy = translate_rotate_point(x, y, theta, translate)
        if xmin <= newx <= xmax and ymin <= newy <= ymax and zmin <= z <= zmax:
            in_bounds.append(event)


    return original_corners, in_bounds, theta, centerx, centery, norm_vec

#EXAMPLE

data = [[1, [0, 0, 0], [20, 3, 45]],
        [3, [3, 5, -2], [40, 10, 60]],
        [5, [1, 20, -4], [10, 30, 40]]]

start = (0, 10)
end = (3, 30)
width = 4
depth = 100
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
corners, in_bounds, theta, centerx, centery, norm_vec = profile_view(data, *start, *end, width, depth)

plot_focal_mechanisms(in_bounds, ax, alpha = 1)
ax.view_init(0, -theta*180/np.pi)
scale_factors = scale_beachballs(in_bounds, ax)
nv = np.array([norm_vec[0], norm_vec[1], 0])


for event in in_bounds:
    radius, center, angles = event
    vecs = vectors(event[2])
    coords = lambert(nv, np.array([0, 0, 1]), vecs)

    fig = plt.figure()
    ax = fig.add_subplot()
    for xy_coords in coords:
        X = []
        Y = []
        for x, y in xy_coords:
            X.append(x)
            Y.append(y)
        ax.plot(X, Y)
    ax.fill(X, Y)
    ax.set_aspect('equal')
plt.show()


        
