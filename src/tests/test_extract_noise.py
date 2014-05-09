from unittest import TestCase
from helpers.config_helpers import parse_config
from prepare_data.extract_noise import extract_noise
from helpers.io_helpers import get_absolute_path

__author__ = 'Daeyun Shin'


class TestExtractNoise(TestCase):
    def test_extract_noise(self):
        config_path = '../../config.yaml'
        print get_absolute_path(config_path)
        config = parse_config(get_absolute_path(config_path))
        input_image_dir = config['paths']['input']['image']
        input_annotation_dir = config['paths']['input']['annotation']
        out_dir = config['paths']['output']['negative sample']

        def callback_temp():
            pass

        extract_noise(input_image_dir, input_annotation_dir, out_dir, callback_temp)
        pass
