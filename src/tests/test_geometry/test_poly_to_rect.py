from unittest import TestCase
import numpy as num
from helpers.geometry_helpers import poly_to_rect
from helpers.io_helpers import pretty_print_exception

__author__ = 'Daeyun Shin'


class TestPolyToRect(TestCase):
    def test_poly_to_rect(self):
        poly = num.array([
            (0, 0),
            (5, 0),
            (5, 10),
            (0, 10),
            (0, 0),
        ])
        rect = poly_to_rect(poly)
        self.assertTupleEqual(rect, (0, 0, 5, 10))

    def test_poly_to_rect_floating_point(self):
        poly = num.array([[417.6, 404.22857143],
                          [829., 404.22857143],
                          [829., 766.],
                          [417.6, 766.],
                          [417.6, 404.228571431]])
        try:
            rect = poly_to_rect(poly)
        except Exception as e:
            pretty_print_exception("Floating point error in poly_to_rect", e)
            self.fail("poly_to_rect() raised Exception unexpectedly.")
