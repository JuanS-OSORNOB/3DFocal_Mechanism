#!/usr/bin/env python3
# Last revised: 08/23/20
# (c) <Juan Sebastián Osorno Bolívar & Amy Teegarden>

import os
import unittest
import re
import inspect
from copy import copy

from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from focmech3d import load_events
from focmech3d.focal_mechanism import plot_focal_mechanisms, FocalMechanism
from focmech3d.plotcoords import fm_quadrant, translate_and_scale
from focmech3d.mpl_plots import plot_focal_mechanism
from focmech3d.datautils import load_data, createpath
from focmech3d.topoprofile import plot_profile
from profile_example import example

directory = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
act_img_path = os.path.join(directory, 'actual_images')
createpath(act_img_path)
exp_img_path = os.path.join(directory, 'expected_images')

def act_img(img_filename):
    return os.path.join(act_img_path, img_filename)
def exp_img(img_filename):
    return os.path.join(exp_img_path, img_filename)
class imgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_list = [[1, [0, 0, -6], [0, 0, 0]],
            [3, [-5, -5, -10], [10, 30, 40]],
            [5, [-5, 10, -8], [280, 90, -30]],
            [2, [5, -5, -10], [90, 0, 100]],
            [10, [20, 20, -10], [359, 45, -100]]]
        cls.data_list2 = [[0, 0, -6, 1, 0, 0, 0], 
                        [-5, -5, -10, 3, 10, 30, 40], 
                        [-5, 10, -8, 5, 280, 90, -30],
                        [5, -5, -10, 2, 90, 0, 100],
                        [20, 20, -10, 10, 359, 45, -100]]
        cls.fms = [FocalMechanism(*data, invert_z = False) for data in cls.data_list2]
    def img_comp(self, fig, img_file, tolerance = .01):
        actual = act_img(img_file)
        fig.savefig(actual)
        diff = compare_images(exp_img(img_file), actual, tolerance)
        plt.close(fig)
        self.assertIsNone(diff)

class test_single_fm(unittest.TestCase):
    def setUp(self):
        self.fregex = re.compile('strike(.+)_dip(.+)_rake(.+).png')
    def test_fms(self):
        for f in os.listdir('tests/expected_images/single_fms/'):
            with self.subTest(f, f = f):
                exp_img = 'tests/expected_images/single_fms/{}'.format(f)
                act_img = 'tests/actual_images/{}'.format(f)
                s, d, r = [int(x) for x in self.fregex.search(f).groups()]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection = '3d')
                fm = FocalMechanism(0, 0, 0, 1, s, d, r)
                plot_focal_mechanism(fm, ax, [1, 1, 1], shade = False)
                fig.savefig(act_img)
                plt.close(fig)
                diff = compare_images(exp_img, act_img, .01)
                self.assertIsNone(diff)

class test_plot_multi(imgTest):

    def setUp(self):
        self.data_list_rad = [[1, [0, 0, -6], [0, 0, 0]],
        [3, [-5, -5, -10], [10 * np.pi/180, 30 * np.pi/180, 40 * np.pi/180]],
        [5, [-5, 10, -8], [280 * np.pi/180, 90 * np.pi/180, -30 * np.pi/180]],
        [2, [5, -5, -10], [90 * np.pi/180, 0, 100 * np.pi/180]],
        [10, [20, 20, -10], [359 * np.pi/180, 45 * np.pi/180, -100 * np.pi/180]]]
        self.data_list_rad = [[0, 0, -6, 1, 0, 0, 0], 
            [-5, -5, -10, 3, 10 * np.pi/180, 30 * np.pi/180, 40 * np.pi/180],
            [-5, 10, -8, 5, 280 * np.pi/180, 90 * np.pi/180, -30 * np.pi/180],
            [5, -5, -10, 2, 90 * np.pi/180, 0, 100 * np.pi/180],
            [20, 20, -10, 10, 359 * np.pi/180, 45 * np.pi/180, -100 * np.pi/180]]
        self.rad_data =  [FocalMechanism(*data, invert_z = False, in_degrees = False) for data in self.data_list_rad]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')

    def test_radians(self):
        plot_focal_mechanisms(self.rad_data, self.ax, in_fms = True, alpha = .5)
        self.img_comp(self.fig, 'rad_test.png', .1)

    def test_vectors(self):
        plot_focal_mechanisms(self.fms, self.ax, vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'T', 'P'], vector_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black'],
        alpha = .5, in_fms = True)
        self.img_comp(self.fig, 'vector_test.png')

class test_event_profile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fig1, fig2 = example(depth_mag = True, verbose = False, show_plots = False)
        fig1.savefig('tests/actual_images/3D_Profile_example.png')
        fig2.savefig('tests/actual_images/Plot_profile_example.png')
    def test_3D_profile(self):
        diff = compare_images('tests/expected_images/3D_Profile_example.png', 'tests/actual_images/3D_Profile_example.png', .01)
        self.assertIsNone(diff)
    def test_2D_profile(self):
        diff = compare_images('tests/expected_images/Plot_profile_example.png', 'tests/actual_images/Plot_profile_example.png', .01)
        self.assertIsNone(diff)

class test_basic_profile(imgTest):
    def test_profile1(self):
        fig = plot_profile(self.fms, [], -10, -10, 50, 50, 20, 40, in_degrees = False, verbose = False, in_fms = True)
        self.img_comp(fig, 'basic_profile1.png')
    def test_profile2(self):
        fig = plot_profile(self.fms, [],  50, 50, -10, -10, 20, 40, in_degrees = False, verbose = False, in_fms = True)
        self.img_comp(fig, 'basic_profile2.png')
        fig.savefig('tests/actual_images/basic_profile2.png')
    def test_profile3(self):
        bigger_fms = []
        for fm in self.fms:
            fm = copy(fm)
            fm.rad_function = lambda x: 20 * x
            fm.radius = fm.radius * 20
            bigger_fms.append(fm)

        fig = plot_profile(bigger_fms, [], -10, 10, 25, 25, 20, 500, in_degrees = True, fm_size = 20, verbose = False, in_fms = True)
        self.img_comp(fig, 'basic_profile3.png')

class test_coords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.numpy_2d_int = np.array([[1, 2, 3, 10, 2], [4, 6, 3, 6, 4]])
        cls.numpy_3d_int = np.array([[3, 4, 6, 4], [2, 3, 5, 6], [1, 1, 2, 4]])
    def test_no_change2d(self):
        coords = translate_and_scale(self.numpy_2d_int, [0, 0], [1, 1])
        self.assertTrue(np.array_equal(coords, self.numpy_2d_int))
    def test_translate2dpos(self):
        coords = translate_and_scale(self.numpy_2d_int, [1, 1])
        expected = np.array([[2, 3, 4, 11, 3], [5, 7, 4, 7, 5]])
        self.assertTrue(np.array_equal(coords, expected))
    def test_translate2dneg(self):
        coords = translate_and_scale(self.numpy_2d_int, [-1, -1])
        expected = np.array([[0, 1, 2, 9, 1], [3, 5, 2, 5, 3]])
        self.assertTrue(np.array_equal(coords, expected))
    def test_scale2dlarger(self):
        coords = translate_and_scale(self.numpy_2d_int, [0, 0], [2, 2])
        expected = np.array([[2, 4, 6, 20, 4], [8, 12, 6, 12, 8]])
        self.assertTrue(np.array_equal(coords, expected))
    def test_full_2d(self):
        coords = translate_and_scale(self.numpy_2d_int, [1, -5], [2, .5])
        expected = np.array([[3, 5, 7, 21, 5], [-3, -2, -3.5, -2, -3]])
        self.assertTrue(np.array_equal(coords, expected))
    def test_no_change3d(self):
        coords = translate_and_scale(self.numpy_3d_int, [0, 0, 0], [1, 1, 1])
        self.assertTrue(np.array_equal(coords, self.numpy_3d_int))
    def test_full_3d(self):
        coords = translate_and_scale(self.numpy_3d_int, [1, -2, 0], [2, .5, 3])
        expected = np.array([[7, 9, 13, 9], [-1, -.5, .5, 1], [3, 3, 6, 12]])
        self.assertTrue(np.array_equal(coords, expected))

class test_top_removed(imgTest):
    def setUp(self):
        self.ax = plot_focal_mechanisms(self.fms, in_fms = True, bottom_half = True)
        self.fig = self.ax.get_figure()
    def test_top_removed_side_view(self):
        self.img_comp(self.fig, 'top_removed1.png')
    def test_top_removed_top_view(self):
        self.ax.view_init(90, 270)
        self.img_comp(self.fig, 'top_removed2.png')

class test_load_data(unittest.TestCase):
    def test_col_order(self):
        data = load_data('tests/test_csv.csv', usecols = [0, 1, 2])
        self.assertTrue(data.columns[0] == 'magnitude')
        data = load_data('tests/test_csv.csv', usecols = [2, 1, 0])
        self.assertTrue(data.columns[0] == 'latitude')
    def test_delimiter(self):
        data = load_data('tests/test_csv_tab.csv', usecols = [0, 1, 2])



class test_load_fms(unittest.TestCase):
    pass
if __name__ == '__main__':
    unittest.main()