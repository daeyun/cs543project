from sklearn.externals import joblib
from helpers.config_helpers import parse_config, parse_annotations
from helpers.io_helpers import get_absolute_path, search_files_by_extension

__author__ = 'Daeyun Shin'

config_path = '../../config.yaml'

config = parse_config(get_absolute_path(config_path))
classifier_path = config['paths']['input']['classifier']
classifier = joblib.load(classifier_path)

source_img_dir = config['paths']['input']['initial']['image']
annotation_dir = config['paths']['input']['initial']['annotation']

annotations = parse_annotations(annotation_dir)
image_paths = search_files_by_extension(source_img_dir, ["JPG", "PNG"])
