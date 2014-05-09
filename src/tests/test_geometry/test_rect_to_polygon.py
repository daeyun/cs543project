from unittest import TestCase
from numpy import testing
from numpy.ma import array
from testing_helpers.geometry_helpers import rect_to_polygon

__author__ = 'Daeyun Shin'


class TestRectToPolygon(TestCase):
    def test_rect_to_polygon(self):
        rects = array([[0, 0, 10, 10], [0, 0, 5, 10], [0, 0, 10, 10]])
        polys = array([
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [(0, 0), (5, 0), (5, 10), (0, 10), (0, 0)],
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        ])
        result = rect_to_polygon(rects)
        testing.assert_almost_equal(polys, result)
