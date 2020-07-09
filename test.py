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
                focal_mechanism(1, [0, 0, 0], [s, d, r], ax, [1, 1, 1])
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

    def save_comp(self, expected, actual):
        self.fig.savefig(actual)
        plt.close(self.fig)
        diff = compare_images(expected, actual, .01)
        self.assertIsNone(diff)



    def test_radians(self):
        plot_focal_mechanisms(self.data_list_rad, self.ax, degrees = False, alpha = .5)
        expected = 'expected_images/rad_test.png'
        actual = 'actual_images/rad_test.png'
        self.save_comp(expected, actual)

    def test_vectors(self):
        plot_focal_mechanisms(self.data_list, self.ax, vector_plots = ['strike', 'dip', 'rake', 'normal', 'B', 'T', 'P'], vector_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black'],
        alpha = .5)
        expected = 'expected_images/vector_test.png'
        actual = 'actual_images/vector_test.png'
        self.save_comp(expected, actual)
    
    def test_quadrant(self):
        expected = np.array((np.array([[-0.7649541 , -0.32943313,  0.2319201 ,  0.70468773,  0.9082886 ,
         0.7649541 ],
       [-0.7649541 , -0.41901223,  0.08697807,  0.5597457 ,  0.8187095 ,
         0.7649541 ],
       [-0.7649541 , -0.5281539 , -0.08961687,  0.38315076,  0.70956782,
         0.7649541 ],
       [-0.7649541 , -0.64617461, -0.28057839,  0.19218924,  0.59154712,
         0.7649541 ],
       [-0.7649541 , -0.76152166, -0.46721383,  0.0055538 ,  0.47620007,
         0.7649541 ],
       [-0.7649541 , -0.86290407, -0.63125402, -0.15848639,  0.37481766,
         0.7649541 ]]), np.array([[-0.5178216 , -0.46996039, -0.24259029,  0.07744106,  0.36789255,
         0.5178216 ],
       [-0.5178216 , -0.31287774,  0.01157478,  0.33160613,  0.52497521,
         0.5178216 ],
       [-0.5178216 , -0.16617588,  0.24894338,  0.56897473,  0.67167707,
         0.5178216 ],
       [-0.5178216 , -0.044215  ,  0.44628022,  0.76631157,  0.79363794,
         0.5178216 ],
       [-0.5178216 ,  0.0410665 ,  0.5842686 ,  0.90429994,  0.87891945,
         0.5178216 ],
       [-0.5178216 ,  0.08132069,  0.64940124,  0.96943259,  0.91917364,
         0.5178216 ]]), np.array([[ 0.38302222,  0.81890845,  0.94199948,  0.70527873,  0.19916547,
        -0.38302222],
       [ 0.38302222,  0.85237097,  0.99614298,  0.75942223,  0.232628  ,
        -0.38302222],
       [ 0.38302222,  0.83272987,  0.96436301,  0.72764226,  0.21298689,
        -0.38302222],
       [ 0.38302222,  0.76190774,  0.8497704 ,  0.61304965,  0.14216477,
        -0.38302222],
       [ 0.38302222,  0.64683716,  0.66358228,  0.42686153,  0.02709418,
        -0.38302222],
       [ 0.38302222,  0.49878203,  0.42402405,  0.1873033 , -0.12096095,
        -0.38302222]])))
        actual = np.array(fm_quadrant(0, [10, 30, 40], True, 6))
        self.assertTrue(np.allclose(actual, expected))

if __name__ == '__main__':
    unittest.main()