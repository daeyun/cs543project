import random
from unittest import TestCase
import cv2
from helpers.geometry_helpers import find_area, line_intersection, is_point_on_line_segment
import numpy as np


class TestLineIntersection(TestCase):
    def test_line_intersection_cross(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(10, 0), (0, 10)]
        x, y = line_intersection(line1, line2)
        self.assertAlmostEqual(x, 5)
        self.assertAlmostEqual(y, 5)

    def test_line_intersection_parallel(self):
        line1 = [(0, 0), (10, 0)]
        line2 = [(5, 5), (5, 0)]
        result = line_intersection(line1, line2)
        self.assertEqual(result, None)

    def test_line_intersection_not_intersecting(self):
        line1 = [(0, 0), (10, 0)]
        line2 = [(5, 5), (7, 1)]
        result = line_intersection(line1, line2)
        self.assertEqual(result, None)

    def test_line_intersection_aligned_not_intersecting(self):
        line1 = [(0, 0), (10, 0)]
        line2 = [(11, 0), (15, 0)]
        result = line_intersection(line1, line2)
        self.assertEqual(result, None)

    def test_line_intersection_aligned_connecting(self):
        line1 = [(0, 0), (11, 0)]
        line2 = [(10, -1), (10, 10)]
        x, y = line_intersection(line1, line2)
        self.assertAlmostEqual(x, 10)
        self.assertAlmostEqual(y, 0)


class TestIsPointOnLineSegment(TestCase):
    def test_vertical(self):
        line = [(0, 0), (0, 10)]
        point = (0, 5)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, True)

    def test_horizontal(self):
        line = [(10, 0), (0, 0)]
        point = (5, 0)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, True)

    def test_diagonal(self):
        line = [(0, 0), (10, 10)]
        point = (6, 6)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, True)

    def test_diagonal_close(self):
        line = [(0, 0), (10, 10)]
        point = (6.01, 6.01)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, True)

    def test_not_on_line(self):
        line = [(0, 0), (10, 10)]
        point = (6, 7)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, False)

    def test_not_on_line_aligned_right(self):
        line = [(0, 0), (10, 10)]
        point = (11, 11)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, False)

    def test_not_on_line_aligned_left(self):
        line = [(0, 0), (10, 10)]
        point = (-1, -1)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, False)

    def test_not_on_line_aligned_bottom(self):
        line = [(0, 0), (0, 10)]
        point = (0, 11)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, False)

    def test_not_on_line_aligned_top(self):
        line = [(0, 0), (10, 0)]
        point = (-1, 0)
        result = is_point_on_line_segment(line, point)
        self.assertEqual(result, False)


class TestFindArea(TestCase):
    def test_find_area_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        area = find_area(points)
        self.assertEqual(20 * 20, area)

    def test_find_area_translated_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        points = np.add(np.array(points), 17)
        area = find_area(points)
        self.assertEqual(20 * 20, area)

    def test_find_area_rescaled_square(self):
        points = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
        points = np.multiply(np.array(points), 0.7)
        points = np.add(np.array(points), 5)
        area = find_area(points)
        self.assertEqual(20 * 20 * 0.7 * 0.7, area)

    def test_find_area_square_reverse(self):
        points = [(0, 0), (0, 10), (10, 10), (10, 0)]
        area = find_area(points)
        self.assertEqual(10 * 10, area)

        area = find_area(points[::-1])
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

        area = find_area(np.array(np.transpose(rotated_points)))
        self.assertAlmostEqual(20 * 20, area)

    def test_find_area_rectangle(self):
        points = [(-10, -10), (5, -10), (5, 10), (-10, 10), (-10, -10)]
        area = find_area(points)
        self.assertAlmostEqual(15 * 20, area)

    def test_find_area_rotated_rectangle(self):
        points = [(-10, -10), (5, -10), (5, 10), (-10, 10), (-10, -10)]
        points_mat = np.vstack((np.transpose(np.matrix(points)), np.ones((1, len(points)))))
        rot_mat = cv2.getRotationMatrix2D((100, 0), 90, 1)
        rotated_points = rot_mat * points_mat
        area = find_area(np.array(np.transpose(rotated_points)))
        self.assertAlmostEqual(15 * 20, area)
