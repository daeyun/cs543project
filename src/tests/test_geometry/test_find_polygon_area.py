import os
from unittest import TestCase
import cv2
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../kv_storage/')
from testing_helpers.geometry_helpers import find_polygon_area

__author__ = 'Daeyun Shin'


class TestFindPolygonArea(TestCase):
    def test_find_area_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        area = find_polygon_area(points)
        self.assertEqual(20 * 20, area)

    def test_find_area_translated_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        points = np.add(np.array(points), 17)
        area = find_polygon_area(points)
        self.assertEqual(20 * 20, area)

    def test_find_area_rescaled_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        points = np.multiply(np.array(points), 0.7)
        points = np.add(np.array(points), 5)
        area = find_polygon_area(points)
        self.assertEqual(20 * 20 * 0.7 * 0.7, area)

    def test_find_area_square_reverse(self):
        points = [(0, 0), (0, 10), (10, 10), (10, 0)]
        area = find_polygon_area(points)
        self.assertEqual(10 * 10, area)

        area = find_polygon_area(points[::-1])
        self.assertEqual(10 * 10, area)

    def test_find_area_rotated_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        points_mat = np.vstack((np.transpose(np.matrix(points)), np.ones((1, len(points)))))

        """ points_mat =
        [[-10.  10.  10. -10. -10.]
         [-10. -10.  10.  10. -10.]
         [  1.   1.   1.   1.   1.]]
        """

        rot_mat = cv2.getRotationMatrix2D((100, 0), 90, 1)
        rotated_points = rot_mat * points_mat

        """ rotated_points =
        [[  90.   90.  110.  110.   90.]
         [ 110.   90.   90.  110.  110.]]
        """

        area = find_polygon_area(np.array(np.transpose(rotated_points)))
        self.assertAlmostEqual(20 * 20, area)

    def test_find_area_rectangle(self):
        points = [(-10, -10), (5, -10), (5, 10), (-10, 10), (-10, -10)]
        area = find_polygon_area(points)
        self.assertAlmostEqual(15 * 20, area)

    def test_find_area_rotated_rectangle(self):
        points = [(-10, -10), (5, -10), (5, 10), (-10, 10), (-10, -10)]
        points_mat = np.vstack((np.transpose(np.matrix(points)), np.ones((1, len(points)))))
        rot_mat = cv2.getRotationMatrix2D((100, 0), 90, 1)
        rotated_points = rot_mat * points_mat
        area = find_polygon_area(np.array(np.transpose(rotated_points)))
        self.assertAlmostEqual(15 * 20, area)
