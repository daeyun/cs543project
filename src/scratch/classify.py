import json
import cv2
from sklearn.externals import joblib
import time
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

classifier = joblib.load('../classifiers/svm-50-100.clf')
classifier2 = joblib.load('../classifiers/svm-300-2000.clf')
feature_extractor = HogObjFeatureExtractor()

detector = SlidingWindow(feature_extractor, classifier, win_size=90, img_size=800, resize_factor=0.94)

# load all annotation files in the given directory
image_manager = ImageManager('./img')

# raises exception if annotation does not exist for this image
image, rect_sets = image_manager.load_annotated_image('./img/DSCF0918.JPG')



# print detector.extract_features(image, rect_sets)

print rect_sets
for rect_set in rect_sets:
    container_rect, label_rects = rect_set
    print container_rect
    print label_rects

    t = time.time()
    positives = detector.detect(image, container_rect)
    print time.time() - t
    print len(positives)
    plot_rects_on_image(image, [container_rect]+positives, ['red', 'green'], thickness=1)
    print positives
    detector.set_classifier(classifier2)
    t = time.time()
    positives = detector.detect(image, container_rect, windows=positives)
    print positives
    print time.time() - t
    t = time.time()
    print len(positives)

    plot_rects_on_image(image, [container_rect]+positives, ['red', 'green'], thickness=2)

    positives = cv2.groupRectangles(positives, 1)
    np = []
    for p in positives[0]:
        print '1',p
        x, y, w, h = p
        np.append((x, y, w, h))
    positives = np
    print positives


    plot_rects_on_image(image, [container_rect]+list(positives), ['red', 'green'], thickness=2)

    exit()