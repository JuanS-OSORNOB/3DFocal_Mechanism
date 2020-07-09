from focal_mechanism import focal_mechanism, plot_focal_mechanisms
from vector_math import fm_quadrant
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images
import re
from mpl_toolkits.mplot3d import Axes3D
import os
import unittest
import numpy as np

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
                focal_mechanism(1, [0, 0, 0], [s, d, r], ax, [1, 1, 1], shade = False)
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
    
if __name__ == '__main__':
    unittest.main()