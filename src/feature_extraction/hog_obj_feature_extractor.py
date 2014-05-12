import cv2
from helpers.feature_extraction_helpers import compute_hog, compute_lab_histogram, get_intersecting_rect, get_mask
from helpers.geometry_helpers import expand_rect
from helpers.image_operation_helpers import crop_image
import numpy as np

__author__ = 'Daeyun Shin'


class HogObjFeatureExtractor:
    """
    Combination of HOG and objectness cues
    """

    def __init__(self):
        self.lab_hist_cache = {}

    def compute_features(self, img, rect, container_rect):
        """
        :type img: ndarray
        :type rect: tuple
        :type container_rect: tuple
        :rtype : ndarray

        :param img: entire source image
        :param rect: ROI (x, y, w, h) to extract features
        :param container_rect: containing border rectangle (x, y, w, h)
        :return: ndarray of shape (135,) of type float32 containing the following features
                 normalized to [0, 1] range.
                   * 128 HOG features
                   * 3   Dissimilarity based on inverse color contrast
                   * 4   Size and location where (0, 0) is the center of container_rect
        """
        roi_img = crop_image(img, rect)

        # HOG feature vector
        hog = compute_hog(roi_img)

        # Color contrasts
        cc = self.compute_surrounding_color_contrast(img, rect, container_rect)
        # In order to fit cc features in [0, 1], use the dissimilarity instead
        cc_dissimilarity = [1 / (i / 10 + 1) for i in cc]

        # Location and size
        ls = self.normalize_location_and_size(rect, container_rect)

        feature_vector = np.hstack((hog, cc_dissimilarity, ls))

        return feature_vector.astype(np.float32)

    @staticmethod
    def normalize_location_and_size(rect, container_rect):
        # normalize x, y, w, h values to be used as features.
        """
        :type rect: tuple
        :param rect: ROI (x, y, w, h) to be used as features
        :param container_rect: containing border rectangle (x, y, w, h) used to normalize rect
        :return: ndarray of shape (4,) containing the normalized x, y, w, h values
        """
        x, y, w, h = rect
        bx, by, bw, bh = container_rect

        # TODO: vectorize this

        # center coordinates
        cx = x + w / 2.0
        cy = y + h / 2.0

        # origin = center of bounding rect
        ox = bx + bw / 2.0
        oy = by + bh / 2.0

        # distance of the center from the origin
        dcx = cx - ox
        dcy = cy - oy

        # normalize to [0, 1] range
        dcx = dcx / bw + 0.5
        dcy = dcy / bh + 0.5
        # size factors. also normalized
        dw = w / bw
        dh = h / bh

        return np.array((dcx, dcy, dw, dh))

    def compute_surrounding_color_contrast(self, img, rect, container_rect, layers=2):
        """
        Measure of the dissimilarity of a window to its immediate surrounding area based on chi-squared
        distances of LAB space histogram. See "Measuring the objectness of image windows" Alexe et al.

        :type img: ndarray
        :type rect: tuple
        :type container_rect: tuple
        :type layers: int
        :rtype : ndarray

        :param img: Source image. This is not a cropped ROI image.
        :param rect: ROI (x, y, w, h) to compute the color contrast against surroundings.
        :param container_rect: Container border rectangle (x, y, w, h) pixels outside this will be ignored.
        :param layers: number of surrounding layers to compute contrasts.
        :return: Color contrast feature vector of shape (layers+1, ). The last cell contains the contrast
                   between theROI and the entire image.
        """

        # chi-squared distances between window image and its surroundings
        cc_feature_vector = np.zeros(layers+1)

        rect_img = crop_image(img, rect)
        window_hist = compute_lab_histogram(rect_img)

        prev_rect = rect
        for level in range(layers):
            expanded_rect = expand_rect(prev_rect, radius_increase=1.5)

            # crop parts outside the container rect
            expanded_rect = get_intersecting_rect(expanded_rect, container_rect)
            expanded_rect_img = crop_image(img, expanded_rect)

            # mask of only the immediate surrounding area
            mask = get_mask(expanded_rect, prev_rect)

            # LAB histogram of the surrounding area
            surrounding_hist = compute_lab_histogram(expanded_rect_img, mask=mask)

            # method=1 is the same as cv.CV_COMP_CHISQR for computing the chi-squared distance
            cc_feature_vector[level] = cv2.compareHist(window_hist, surrounding_hist, method=1)

            prev_rect = expanded_rect

        # cache the background histogram to save computation time
        img_id = id(img)
        if img_id not in self.lab_hist_cache:
            rect_img = crop_image(img, container_rect)
            background_hist = compute_lab_histogram(rect_img)
            self.lab_hist_cache[img_id] = background_hist
        else:
            background_hist = self.lab_hist_cache[img_id]

        # color contract between the window image and the entire source image
        background_hist_dist = cv2.compareHist(window_hist, background_hist, method=1)
        cc_feature_vector[layers] = background_hist_dist

        return cc_feature_vector