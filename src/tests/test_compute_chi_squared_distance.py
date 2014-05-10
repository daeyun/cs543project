from unittest import TestCase
import numpy as num
from helpers.feature_extraction_helpers import chi_squared_distance

__author__ = 'Daeyun Shin'


class TestComputeChiSquaredDistance(TestCase):
    def test_compute_chi_squared_distance(self):
        """
        Compute d(x,y) = sum((xi-yi)^2/(xi+yi)) / 2 between two histograms.
        """
        a = num.random.rand(4, 4, 8)
        b = num.random.rand(4, 4, 8)
        dist = chi_squared_distance(a, b)

        sq_sum = 0
        a_flat = a.flatten()
        b_flat = b.flatten()
        for i in range(a_flat.shape[0]):
            sq_sum += ((a_flat[i] - b_flat[i]) ** 2.0)/(a_flat[i] + b_flat[i])

        self.assertAlmostEqual(sq_sum/2, dist)