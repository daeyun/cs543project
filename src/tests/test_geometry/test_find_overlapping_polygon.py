from unittest import TestCase
from cv2 import getRotationMatrix2D
from numpy import transpose, vstack, ones
from numpy.ma import dot
from numpy.matrixlib import matrix
from helpers.geometry_helpers import find_overlapping_polygon, check_polygon_equality, find_polygon_area
from helpers.image_processing_helpers import getTranslationMatrix2d

__author__ = 'Daeyun Shin'


class TestFindOverlappingPolygon(TestCase):
    def test_two_overlapping_squares(self):
        poly1 = matrix([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])
        dx, dy = (5, 5)
        trans_mat = getTranslationMatrix2d(dx, dy)
        homogeneous_trans_mat = dot(trans_mat, vstack((transpose(poly1), ones((1, poly1.shape[0])))))
        poly2 = transpose(homogeneous_trans_mat[0:2, :])

        intersection = matrix([(5, 5), (5, 10), (10, 10), (10, 5), (5, 5), ])
        new_poly = find_overlapping_polygon(poly1, matrix(poly2))
        self.assertNotEqual(new_poly, None)
        self.assertTupleEqual(new_poly.shape, (5, 2))
        result = check_polygon_equality(new_poly, intersection)
        self.assertTrue(result)

    def test_two_non_overlapping_squares(self):
        poly1 = matrix([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])
        dxs = [-10, 10, 0]
        dys = [-10, 10, 0]
        for dx in dxs:
            for dy in dys:
                if (dx, dy) == (0, 0):
                    continue

                trans_mat = getTranslationMatrix2d(dx, dy)
                homogeneous_trans_mat = dot(trans_mat, vstack((transpose(poly1), ones((1, poly1.shape[0])))))
                poly2 = transpose(homogeneous_trans_mat[0:2, :])

                new_poly = find_overlapping_polygon(poly1, matrix(poly2))
                self.assertEqual(new_poly, None)

    def test_two_non_overlapping_squares(self):
        poly1 = matrix([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])
        dxys = [(-8, 0), (0, 8), (8, 0), (0, -8)]

        for dx, dy in dxys:
            if (dx, dy) == (0, 0):
                continue

            trans_mat = getTranslationMatrix2d(dx, dy)
            homogeneous_trans_mat = dot(trans_mat, vstack((transpose(poly1), ones((1, poly1.shape[0])))))
            poly2 = transpose(homogeneous_trans_mat[0:2, :])

            new_poly = find_overlapping_polygon(poly1, matrix(poly2))
            area = find_polygon_area(new_poly)
            self.assertAlmostEqual(area, 2 * 10)

    def test_contained_square(self):
        poly1 = matrix([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])
        poly2 = matrix([(-10, -10), (20, -10), (20, 20), (-10, 20), (-10, -10), ])
        new_poly = find_overlapping_polygon(poly1, poly2)

        result = check_polygon_equality(new_poly, poly1)
        self.assertTrue(result)

    def test_rotated_squares(self):
        poly1 = matrix([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])

        rot_mat = getRotationMatrix2D((5, 5), 45, 1.0)
        homogeneous_poly1 = vstack((transpose(poly1), ones((1, poly1.shape[0]))))

        poly2 = transpose(dot(rot_mat, homogeneous_poly1))

        new_poly = find_overlapping_polygon(poly1, matrix(poly2))

        ## plot the result
        # import matplotlib.pyplot as plt
        # from matplotlib.path import Path
        # import matplotlib.patches as patches
        # path = Path(new_poly)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # patch = patches.PathPatch(path, facecolor='orange', lw=2)
        # ax.add_patch(patch)
        # ax.set_xlim(-5,15)
        # ax.set_ylim(-5,15)
        # plt.show()

        # should be an octagon
        self.assertEqual(new_poly.shape[0], 9)