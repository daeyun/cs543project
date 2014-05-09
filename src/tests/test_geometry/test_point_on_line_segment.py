from unittest import TestCase
from testing_helpers.geometry_helpers import is_point_on_line_segment

__author__ = 'Daeyun Shin'


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