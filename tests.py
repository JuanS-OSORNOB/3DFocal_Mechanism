from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
from topoprofile import plot_profile
import re
from mpl_toolkits.mplot3d import Axes3D
import os
import unittest
import numpy as np
from profile_example import example
from focal_mechanism import focal_mechanism, plot_focal_mechanisms, FocalMechanism
from plotcoords import fm_quadrant, translate_and_scale

class test_single_fm(unittest.TestCase):
    def setUp(self):
        self.fregex = re.compile('strike(.+)_dip(.+)_rake(.+).png')
        if not os.path.isdir('actual_images'):
            os.mkdir('actual_images')
    def test_fms(self):
        for f in os.listdir('expected_images/single_fms/'):
            with self.subTest(f, f = f):
                exp_img = 'expected_images/single_fms/{}'.format(f)
                act_img = 'actual_images/{}'.format(f)
                s, d, r = [int(x) for x in self.fregex.search(f).groups()]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection = '3d')
                fm = FocalMechanism(0, 0, 0, 1, s, d, r)
                focal_mechanism(fm, ax, [1, 1, 1], shade = False)
                fig.savefig(act_img)
                plt.close(fig)
                diff = compare_images(exp_img, act_img, .01)
                self.assertIsNone(diff)

class test_plot_multi(unittest.TestCase):

    def setUp(self):
        self.data_list = [[1, [0, 0, 0], [0, 0, 0]],
            [3, [-5, -5, -5], [10, 30, 40]],
            [5, [-5, 10, 0], [280, 90, -30]],
            [2, [5, -5, -5], [90, 0, 100]],
            [10, [20, 20, -20], [359, 45, -100]]]
        self.data_list_rad = [[1, [0, 0, 0], [0, 0, 0]],
            [3, [-5, -5, -5], [10 * np.pi/180, 30 * np.pi/180, 40 * np.pi/180]],
            [5, [-5, 10, 0], [280 * np.pi/180, 90 * np.pi/180, -30 * np.pi/180]],
            [2, [5, -5, -5], [90 * np.pi/180, 0, 100 * np.pi/180]],
            [10, [20, 20, -20], [359 * np.pi/180, 45 * np.pi/180, -100 * np.pi/180]]]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')
        if not os.path.isdir('actual_images'):
            os.mkdir('actual_images')

    def save_comp(self, expected, actual, tolerance = .01):
        self.fig.savefig(actual)
        plt.close(self.fig)
        diff = compare_images(expected, actual, tolerance)
        self.assertIsNone(diff)

    def test_radians(self):
        plot_focal_mechanisms(self.data_list_rad, self.ax, degrees = False, alpha = .5)
        expected = 'expected_images/rad_test.png'
        actual = 'actual_images/rad_test.png'
        self.save_comp(expected, actual, .1)

    def test_vectors(self):
        plot_focal_mechanisms(self.data_list, self.ax, vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'T', 'P'], vector_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black'],
        alpha = .5)
        expected = 'expected_images/vector_test.png'
        actual = 'actual_images/vector_test.png'
        self.save_comp(expected, actual)

class test_event_profile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fig1, fig2 = example(depth_mag = True, verbose = False, show_plots = False)
        fig1.savefig('actual_images/3D Profile example.png')
        fig2.savefig('actual_images/Plot profile example.png')
    def test_3D_profile(self):
        diff = compare_images('expected_images/3D Profile example.png', 'actual_images/3D Profile example.png', .01)
        self.assertIsNone(diff)
    def test_2D_profile(self):
        diff = compare_images('expected_images/Plot profile example.png', 'actual_images/Plot profile example.png', .01)
        self.assertIsNone(diff)

class test_basic_profile(unittest.TestCase):
    def setUp(self):
        self.data_list = [[1, [0, 0, -6], [0, 0, 0]],
            [3, [-5, -5, -10], [10, 30, 40]],
            [5, [-5, 10, -8], [280, 90, -30]],
            [2, [5, -5, -10], [90, 0, 100]],
            [10, [20, 20, -10], [359, 45, -100]]]
    def test_profile1(self):
        fig = plot_profile(self.data_list, [], -10, -10, 50, 50, 20, 40, in_degrees = False, verbose = False)
        fig.savefig('actual_images/basic_profile1.png')
        diff = compare_images('actual_images/basic_profile1.png', 'expected_images/basic_profile1.png', .01)
        self.assertIsNone(diff)
    def test_profile2(self):
        fig = plot_profile(self.data_list, [],  50, 50, -10, -10, 20, 40, in_degrees = False, verbose = False)
        fig.savefig('actual_images/basic_profile2.png')
        diff = compare_images('actual_images/basic_profile2.png', 'expected_images/basic_profile2.png', .01)
        self.assertIsNone(diff)
    def test_profile3(self):
        fig = plot_profile(self.data_list, [], -10, 10, 25, 25, 20, 500, in_degrees = True, fm_size = 20, verbose = False)
        fig.savefig('actual_images/basic_profile3.png')
        diff = compare_images('actual_images/basic_profile3.png', 'expected_images/basic_profile3.png', .01)
        self.assertIsNone(diff)

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
    
if __name__ == '__main__':
    unittest.main()