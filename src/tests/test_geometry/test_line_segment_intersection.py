from unittest import TestCase
from helpers.geometry_helpers import line_segment_intersection
from tests.helpers.assertion_helpers import assertTupleAlmostEqual

__author__ = 'Daeyun Shin'


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