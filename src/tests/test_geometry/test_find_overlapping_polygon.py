from unittest import TestCase
from numpy import transpose, vstack, ones
from numpy.ma import dot
from numpy.matrixlib import matrix
from helpers.geometry_helpers import find_overlapping_polygon, check_polygon_equality
from helpers.image_processing_helpers import getTranslationMatrix2d

__author__ = 'Daeyun Shin'


class TestFindOverlappingPolygon(TestCase):
    def test_two_overlapping_squares(self):
        poly1 = matrix([
            (0, 0),
            (10, 0),
            (10, 10),
            (0, 10),
            (0, 0),
        ])
        dx, dy = (5, 5)
        trans_mat = getTranslationMatrix2d(dx, dy)
        homogeneous_trans_mat = dot(trans_mat, vstack((transpose(poly1), ones((1, poly1.shape[0])))))
        poly2 = transpose(homogeneous_trans_mat[0:2, :])

        intersection = matrix([
            (5, 5),
            (5, 10),
            (10, 10),
            (10, 5),
            (5, 5),
        ])
        new_poly = find_overlapping_polygon(poly1, matrix(poly2))
        self.assertNotEqual(new_poly, None)
        self.assertTupleEqual(new_poly.shape, (5, 2))
        result = check_polygon_equality(new_poly, intersection)
        self.assertTrue(result)