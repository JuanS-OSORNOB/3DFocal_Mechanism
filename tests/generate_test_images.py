import os
import unittest
import numpy as np
import re
from copy import copy

from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from mpl_toolkits.mplot3d import Axes3D

from focmech3d.mpl_plots import plot_focal_mechanism
from focmech3d.focal_mechanism import plot_focal_mechanisms, FocalMechanism
from focmech3d.topoprofile import plot_profile, profile_view
from profile_example import example

fig1, fig2 = example(depth_mag=True, verbose = False, show_plots = False)
fig1.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/3D_Profile_example.png')
fig2.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/Plot_profile_example.png')

# strike = [0, 10, 180, 360]
# dip = [0, 10, 45, 90]
# rake = [-180, 0, 10, 180]

# fregex = re.compile('strike(.+)_dip(.+)_rake(.+)')

# for s in strike:
#     for d in dip:
#         for r in rake:
#             filename = 'strike{}_dip{}_rake{}'.format(s, d, r)
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection = '3d')
#             fm = FocalMechanism(0, 0, 0, 1, s, d, r)
#             plot_focal_mechanism(fm, ax, shade = False)
#             fig.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/basic_single_fms/{}.png'.format(filename))
#             plt.close(fig)



data_list_old = [[1, [0, 0, -6], [0, 0, 0]],
            [3, [-5, -5, -10], [10, 30, 40]],
            [5, [-5, 10, -8], [280, 90, -30]],
            [2, [5, -5, -10], [90, 0, 100]],
            [10, [20, 20, -10], [359, 45, -100]]]

data_list = [[0, 0, -6, 1, 0, 0, 0], 
[-5, -5, -10, 3, 10, 30, 40], 
[-5, 10, -8, 5, 280, 90, -30],
[5, -5, -10, 2, 90, 0, 100],
[20, 20, -10, 10, 359, 45, -100]]
focal_mechanisms = [FocalMechanism(*data, invert_z = False) for data in data_list]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax = plot_focal_mechanisms(focal_mechanisms, ax, in_fms = True, bottom_half = True)
fig.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/top_removed1.png')
ax.view_init(90, 270)
fig.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/top_removed2.png')

big_focal_mechanisms = []
for fm in focal_mechanisms:
        fm = copy(fm)
        fm.radius = 20 * fm.radius
        fm.rad_function = lambda x: 20 * x
        big_focal_mechanisms.append(fm)

fig1 = plot_profile(focal_mechanisms, [], -10, -10, 50, 50, 20, 40, in_degrees = False, verbose = False, in_fms = True)
fig2 = plot_profile(focal_mechanisms, [],  50, 50, -10, -10, 20, 40, in_degrees = False, verbose = False, in_fms = True)
fig3 = plot_profile(big_focal_mechanisms, [], -10, 10, 25, 25, 20, 500, in_degrees = True, fm_size = 20, verbose = False, in_fms = True)

fig1.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/basic_profile1.png')
fig2.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/basic_profile2.png')
fig3.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/basic_profile3.png')


fig = plt.figure()
data_list = [[1, [0, 0, -6], [0, 0, 0]],
            [3, [-5, -5, -10], [10, 30, 40]],
            [5, [-5, 10, -8], [280, 90, -30]],
            [2, [5, -5, -10], [90, 0, 100]],
            [10, [20, 20, -10], [359, 45, -100]]]
ax = fig.add_subplot(111, projection = '3d')
plot_focal_mechanisms(focal_mechanisms, ax, vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'T', 'P'], 
                        vector_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black'],
                        alpha = .5, in_fms = True)

fig.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/vector_test.png')

data_list_rad = [[1, [0, 0, -6], [0, 0, 0]],
        [3, [-5, -5, -10], [10 * np.pi/180, 30 * np.pi/180, 40 * np.pi/180]],
        [5, [-5, 10, -8], [280 * np.pi/180, 90 * np.pi/180, -30 * np.pi/180]],
        [2, [5, -5, -10], [90 * np.pi/180, 0, 100 * np.pi/180]],
        [10, [20, 20, -10], [359 * np.pi/180, 45 * np.pi/180, -100 * np.pi/180]]]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
plot_focal_mechanisms(focal_mechanisms, ax, in_fms = True, alpha = .5)
fig.savefig('/home/amy/3DFocal_Mechanism/tests/expected_images/rad_test.png')

# angles = [359, 45, -100]
# rad_angles = [np.pi * x / 180 for x in angles]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# focal_mechanism(10, [20, 20, -20], [359, 45, -100], ax, [1, 1, 1])

# fig.savefig('deg.png')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# focal_mechanism(10, [20, 20, -20], [359 * np.pi/180, 45 * np.pi/180, -100 * np.pi/180], ax, [1, 1, 1], degrees = False)
# fig.savefig('rad.png')

# print(compare_images('deg.png', 'rad.png', .01))
