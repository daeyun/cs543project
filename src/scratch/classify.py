import json
import cv2
from sklearn.externals import joblib
from feature_extractor.hog_obj_feature_extractor import HogObjFeatureExtractor
from helpers.config_helpers import parse_annotations
from helpers.feature_extraction_helpers import get_intersecting_rect
from helpers.image_manager import ImageManager
from helpers.image_operation_helpers import rotate_image
from helpers.plotting_helpers import plot_rects_on_image, plot_rect_sets
from sliding_window.sliding_window_detector import SlidingWindow

__author__ = 'Daeyun Shin'

# config_path = '../../config.yaml'

# config = parse_config(get_absolute_path(config_path))
# classifier_path = config['paths']['input']['classifier']
# classifier = joblib.load(classifier_path)

# source_img_dir = config['paths']['input']['initial']['image']
# annotation_dir = config['paths']['input']['initial']['annotation']
#
# annotations = parse_annotations(annotation_dir)
# image_paths = search_files_by_extension(source_img_dir, ["JPG", "PNG"])

# for image_path in image_paths:
#     image = cv2.imread(image_path)
#
#     image_filename = path_to_filename(image_path)
#
#     try:
#         annotation = annotations[image_filename]
#         image_info = annotation['image info']
#     except:
#         print "annotation is not available for {}".format(image_filename)
#         continue
#
#     (img_w, img_h) = image.shape[1], image.shape[0]
#     if (img_w, img_h) != (image_info['w'], image_info['h']):
#         image = cv2.resize(image, (image_info['w'], image_info['h']))
#
#         orientation = image_info['orientation']
#         rotated_image = rotate_image(orientation, 0)
#
#         positive_rects = []
#         border_rects = None
#         image_rect = [-image.shape[1]/2, -image.shape[0]/2, image.shape[1], image.shape[0]]
#         for k, v in annotation['rects'].items():
#             if k == 'border':
#                 border_rects = v
#             else:
#                 positive_rects += v
#
#         if border_rects is None:
#             print 'border rectangle must exist.'
#             continue

classifier = joblib.load('./classifiers/svm.clf')
feature_extractor = HogObjFeatureExtractor()

detector = SlidingWindow(feature_extractor, classifier, win_size=90, img_size=900, resize_factor=0.9)

# load all annotation files in the given directory
image_manager = ImageManager('./img')

# raises exception if annotation does not exist for this image
image, rect_sets = image_manager.load_annotated_image('./img/DSCF0876.JPG')



print detector.extract_features(image, rect_sets)