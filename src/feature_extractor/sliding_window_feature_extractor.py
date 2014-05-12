import os
import uuid
from sklearn.externals import joblib
from feature_extractor.hog_obj_feature_extractor import HogObjFeatureExtractor
from helpers.image_manager import ImageManager
from helpers.io_helpers import search_files_by_extension, pretty_print_exception, path_to_filename, make_sure_dir_exists
from sliding_window.sliding_window_detector import SlidingWindow
import numpy as np

__author__ = 'Daeyun Shin'


class FeatureExtractor:
    def __init__(self, annotation_dir_path):
        # The classifier is not used in this function, but right now it is inconveniently required as a parameter. To be fixed.
        self.classifier = joblib.load('./classifiers/svm.clf')
        self.feature_extractor = HogObjFeatureExtractor()
        self.sliding_window = SlidingWindow(self.feature_extractor, self.classifier, win_size=90, img_size=900, resize_factor=0.9)

        # load all annotation files in the given directory
        self.image_manager = ImageManager(annotation_dir_path)

    def load_images_and_save_features(self, image_dir_path, out_dir):
        image_paths = search_files_by_extension(image_dir_path, ["JPG", "PNG"])
        for image_path in image_paths:
            try:
                # raises exception if annotation does not exist for this image
                image, rect_sets = self.image_manager.load_annotated_image(image_path)
            except Exception as e:
                pretty_print_exception("Annotation was not found for {}".format(image_path), e)
                continue
            try:
                features = self.sliding_window.extract_features(image, rect_sets)
            except Exception as e:
                pretty_print_exception("ERROR. feature extraction failed {}".format(image_path), e)
                continue

            self.save_features_to_file(features, out_dir, image_path)

    def save_features_to_file(self, features, out_dir, source_image_path):
        rand_id = uuid.uuid4().hex[:6]
        source_filename = path_to_filename(source_image_path)
        out_filename = "{}_{}.txt".format(source_filename, rand_id)

        make_sure_dir_exists(out_dir)
        out_path = os.path.join(out_dir, out_filename)

        if features is None:
            print "ERROR: no feature detected? {}".format(source_filename)
            return
        np.savetxt(out_path, features, fmt='%1.14e', delimiter=',')

        print 'saved {} as {}'.format(source_filename, out_path)
