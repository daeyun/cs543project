from unittest import TestCase
from helpers.config_helpers import unpack_filename

__author__ = 'Daeyun Shin'


class TestUnpackFilename(TestCase):
    def test_unpack_filename(self):
        filename = 'DSCF0631.JPG__640_1120_100_100_-1_0.JPG'
        info = unpack_filename(filename)
        self.assertTupleEqual(info, ('DSCF0631.JPG', 640, 1120, 100, 100, -1, 0))

    def test_unpack_filename_absolute_path(self):
        filename = '/User/daeyun/DSCF0631.JPG__640_1120_100_100_-1_0.JPG'
        info = unpack_filename(filename)
        self.assertTupleEqual(info, ('DSCF0631.JPG', 640, 1120, 100, 100, -1, 0))

    def test_unpack_filename_relative_path(self):
        filename = '../DSCF0631.JPG__640_1120_100_100_-1_0.JPG'
        info = unpack_filename(filename)
        self.assertTupleEqual(info, ('DSCF0631.JPG', 640, 1120, 100, 100, -1, 0))
