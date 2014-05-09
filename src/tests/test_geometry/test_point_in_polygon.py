from unittest import TestCase
from testing_helpers.geometry_helpers import is_point_in_polygon

__author__ = 'Daeyun Shin'


class TestIsPointInPolygon(TestCase):
    def test_point_in_square(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        point = (5, 5)
        result = is_point_in_polygon(poly, point)
        self.assertTrue(result)

    def test_point_not_in_square(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        point = (10, -1)
        result = is_point_in_polygon(poly, point)
        self.assertFalse(result)

    def test_point_on_square_border(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        point = (5, 10)
        result = is_point_in_polygon(poly, point)
        self.assertTrue(result)

    def test_point_in_triangle(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 0)]
        point = (5, 3)
        result = is_point_in_polygon(poly, point)
        self.assertTrue(result)

    def test_point_on_triangle_border(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 0)]
        point = (5, 5)
        result = is_point_in_polygon(poly, point)
        self.assertTrue(result)

    def test_point_not_in_triangle(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 0)]
        point = (5.5, 6)
        result = is_point_in_polygon(poly, point)
        self.assertFalse(result)