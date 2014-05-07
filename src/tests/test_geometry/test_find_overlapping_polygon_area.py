from unittest import TestCase
from numpy import array
from helpers.geometry_helpers import find_overlapping_polygon_area

__author__ = 'Daeyun Shin'


class TestFindOverlappingPolygonArea(TestCase):
    def test_contained_square(self):
        poly1 = array([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])
        poly2 = array([(-10, -10), (20, -10), (20, 20), (-10, 20), (-10, -10), ])
        area = find_overlapping_polygon_area(poly1, poly2)
        self.assertAlmostEqual(10*10, area)

    def test_contained_square(self):
        poly1 = array([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ])
        poly2 = array([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), ]) + 5
        area = find_overlapping_polygon_area(poly1, poly2)
        self.assertAlmostEqual(5*5, area)