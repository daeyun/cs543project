from unittest import TestCase
from numpy import array
from testing_helpers.geometry_helpers import rotate_rects
from numpy import testing

__author__ = 'Daeyun Shin'


class TestRotateRects(TestCase):
    def test_rotate_rects(self):
        rects = array([
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        ])
        center = (10, 10)
        angle = 90
        result = rotate_rects(rects, center, angle)
        rotated_rects = array([
            [(0, 20), (0, 10), (10, 10), (10, 20), (0, 20)],
            [(0, 20), (0, 10), (10, 10), (10, 20), (0, 20)],
        ])
        testing.assert_almost_equal(rotated_rects, result)