from unittest import TestCase
from helpers.geometry_helpers import expand_rect

__author__ = 'Daeyun Shin'


class TestExpandRect(TestCase):
    def test_expand_rect_0_origin(self):
        rect = (-10, -10, 10, 10)
        inc = 1.5
        x, y, w, h = expand_rect(rect, radius_increase=inc)
        self.assertAlmostEqual(x, -12)
        self.assertAlmostEqual(y, -12)
        self.assertAlmostEqual(w, 15)
        self.assertAlmostEqual(h, 15)

    def test_expand_rect(self):
        rect = (0, 0, 96, 96)
        inc = 1.5
        x, y, w, h = expand_rect(rect, radius_increase=inc)
        self.assertAlmostEqual(x, int(0-96*0.5/2))
        self.assertAlmostEqual(y, int(0-96*0.5/2))
        self.assertAlmostEqual(w, 96*inc)
        self.assertAlmostEqual(h, 96*inc)
