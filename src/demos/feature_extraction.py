import cv2
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte, color
from skimage.exposure import exposure
from skimage.feature import hog
import numpy as np
from helpers.config_helpers import parse_config, parse_annotations
from helpers.feature_extraction_helpers import compute_hog, compute_lab_histogram
from helpers.io_helpers import get_absolute_path, search_files_by_extension

config_path = '../../config.yaml'
config = parse_config(get_absolute_path(config_path))

source_img_dir = config['paths']['input']['initial']['image']
annotation_dir = config['paths']['input']['initial']['annotation']

positive_set_dir = config['paths']['input']['step one']['positive']
negative_set_dir = config['paths']['input']['step one']['negative']

annotations = parse_annotations(annotation_dir)
positive_set_paths = search_files_by_extension(positive_set_dir, ["JPG", "PNG"])
negative_set_paths = search_files_by_extension(negative_set_dir, ["JPG", "PNG"])

for idx, filename in enumerate(positive_set_paths + negative_set_paths):
    if idx < len(positive_set_paths):
        label = 1
    else:
        label = 0

    img = cv2.imread(filename)
    fd = compute_hog(img)
    # print fd

    compute_lab_histogram(img)