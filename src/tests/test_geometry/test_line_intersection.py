from unittest import TestCase
from helpers.geometry_helpers import line_intersection
from tests.helpers.assertion_helpers import assertTupleAlmostEqual

__author__ = 'Daeyun Shin'


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