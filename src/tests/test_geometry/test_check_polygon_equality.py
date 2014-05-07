from unittest import TestCase
from numpy import vstack
from helpers.geometry_helpers import check_polygon_equality

__author__ = 'Daeyun Shin'


class TestCheckPolygonEquality(TestCase):
    def shift_list(self, l, n):
        """ Shift list to the left """
        return l[n:] + l[:n]

    def test_shifted_points(self):
        poly1 = [(0, 0), (0, 4), (1, 4), (7, 7)]
        for i in range(5):
            poly2 = self.shift_list(poly1, i)
            poly1_enclosed = vstack((poly1, poly1[0]))
            poly2_enclosed = vstack((poly2, poly2[0]))
            result = check_polygon_equality(poly1_enclosed, poly2_enclosed)
            self.assertTrue(result)

    def test_shifted_points_unequal(self):
        poly1 = [(0, 0), (0, 4), (1, 4), (7, 7)]
        poly2 = [(0, 4), (0, 0), (1, 4), (7, 7)]
        for i in range(5):
            poly2_shifted = self.shift_list(poly2, i)
            poly1_enclosed = vstack((poly1, poly1[0]))
            poly2_enclosed = vstack((poly2_shifted, poly2_shifted[0]))
            result = check_polygon_equality(poly1_enclosed, poly2_enclosed)
            self.assertFalse(result)

    def test_almost_equal_points(self):
        poly1 = [(0, 0), (0, 4), (1, 4.00001), (7, 7)]
        poly2 = [(0, 0.000001), (0, 4), (1, 4), (7, 7)]
        for i in range(5):
            poly2_shifted = self.shift_list(poly2, i)
            poly1_enclosed = vstack((poly1, poly1[0]))
            poly2_enclosed = vstack((poly2_shifted, poly2_shifted[0]))
            result = check_polygon_equality(poly1_enclosed, poly2_enclosed)
            self.assertTrue(result)

    def test_almost_equal_points_reverse(self):
        poly1 = [(0, 0), (0, 4), (1, 4.00001), (7, 7)]
        poly2 = [(0, 0.000001), (0, 4), (1, 4), (7, 7)]
        poly2.reverse()
        for i in range(5):
            poly2_shifted = self.shift_list(poly2, i)
            poly1_enclosed = vstack((poly1, poly1[0]))
            poly2_enclosed = vstack((poly2_shifted, poly2_shifted[0]))
            result = check_polygon_equality(poly1_enclosed, poly2_enclosed)
            self.assertTrue(result)