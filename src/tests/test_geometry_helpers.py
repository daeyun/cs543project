import random
from unittest import TestCase
import cv2
import numpy as np
from src.helpers.geometry_helpers import line_intersection, is_point_on_line_segment, find_area, \
    line_segment_intersection
from tests.helpers.assertion_helpers import assertTupleAlmostEqual


class TestLineIntersection(TestCase):
    def test_line_intersection_perpendicular_bottom(self):
        line1 = [(0, 0), (10, 0)]
        line2 = [(5, 10), (5, 2)]
        p = line_intersection(line1, line2)
        assertTupleAlmostEqual(p, (5, 0))

    def test_line_intersection_perpendicular_top(self):
        line1 = [(0, 0), (10, 0)]
        line2 = [(5, -10), (5, -1)]
        p = line_intersection(line1, line2)
        assertTupleAlmostEqual(p, (5, 0))

    def test_line_intersection_perpendicular_top_different_order(self):
        line1 = [(5, -10), (5, -1)]
        line2 = [(10, 0), (0, 0)]
        p = line_intersection(line1, line2)
        assertTupleAlmostEqual(p, (5, 0))

    def test_line_intersection_perpendicular_crossing(self):
        line1 = [(0, 0), (10, 0)]
        line2 = [(5, -10), (5, 2)]
        p = line_intersection(line1, line2)
        assertTupleAlmostEqual(p, (5, 0))

    def test_line_intersection_diagonal(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(0, 10), (2, 8)]
        p = line_intersection(line1, line2)
        assertTupleAlmostEqual(p, (5, 5))


class TestLineSegmentIntersection(TestCase):
    def test_not_crossing(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(0, 10), (2, 8)]
        p = line_segment_intersection(line1, line2)
        self.assertEqual(p, None)

    def test_connecting(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(0, 10), (5, 5)]
        p = line_segment_intersection(line1, line2)
        self.assertEqual(p, (5, 5))

    def test_crossing(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(0, 10), (10, 0)]
        p = line_segment_intersection(line1, line2)
        self.assertEqual(p, (5, 5))

    def test_crossing(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(0, 10), (10, 0)]
        p = line_segment_intersection(line1, line2)
        self.assertTupleAlmostEqual(p, (5, 5))

    def test_perpendicular(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(5, 5), (5, 2)]
        p = line_segment_intersection(line1, line2)
        self.assertEqual(p, None)

    def test_perpendicular_crossing(self):
        line1 = [(0, 0), (10, 10)]
        line2 = [(5, 5), (5, -1)]
        p = line_segment_intersection(line1, line2)
        assertTupleAlmostEqual(p, (5, 5))


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
