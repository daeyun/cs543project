from multiprocessing import Pool, Process
import cv2
import itertools
import time
from helpers.feature_extraction_helpers import get_intersecting_rect2
from helpers.image_operation_helpers import crop_image
from helpers.parallelization_helpers import chunks
from helpers.plotting_helpers import plot_rects_on_image
import numpy as np
import random

__author__ = 'Daeyun Shin'


class SlidingWindow:
    """
    Multi-scale, square sliding window detector
    """

    def __init__(self, feature_extractor, classifier, win_size=90, img_size=700, n_processes=4, resize_factor=0.8):
        """
        :param feature_extractor: A feature extractor object that computes features given a source image,
            ROI rect, and a container rect. It should have the following interface:
            feature_extractor.compute_features(image: ndarray, rect: (x, y, w, w), container_rect: (x, y, w, h)) -> ndarray
        :param classifier: A binary classifier. It should have the following interface:
            self.classifier.predict(features: ndarray) -> 0 or 1
        :param win_size: Size s of the s by s square window.
        :param img_size: Input image will be resized such that the length of the longer side is equal to img_size.
        :param n_processes: Number of processes used to parallelize the workload.
        :param resize_factor: Ratio indicating the image size differences in the image pyramid.
        """
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.image_cache = {}
        self.resize_factor = resize_factor
        self.win_size = win_size
        self.image_pyramid = {}

        self.img_size = img_size
        self.n_processes = n_processes

    def __detector(self, img, container_rect, windows):
        """
        :type img: ndarray
        :type container_rect: tuple
        :type windows: list
        :rtype: list

        :param img: Source image. Ideally a view instead of a copy.
        :param container_rect: Rectangular area (x, y, w, h) to run the sliding window. This parameter is necessary
            because the feature extractor needs relative location and size information.
        :param windows: List of ((x, y), resize level)
        :return: List of (x, y, w, h)
        """
        # count = 0
        print len(windows), ' windows'
        positives = []
        for window in windows:
            (x, y), level = window
            pyr_image = self.get_pyramid_image(img, level)
            rect = (x, y, self.win_size, self.win_size)
            new_container_rect = self.downscale_int_tuple(container_rect, level)

            # if count % 40 == 0 or (level > 0 and count % 2) or level > 2:
            #     print count, level, self.win_size, x, y, pyr_image.shape, new_container_rect
            #     im = pyr_image.copy()
            #     cv2.rectangle(im, (x, y), (x + self.win_size, y + self.win_size), (0, 255, 0), 3)
            #     cx, cy, cw, ch = new_container_rect
            #     cv2.rectangle(im, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 3)
            #     cv2.imshow('Sliding window', im)
            #     cv2.waitKey(0)
            # count += 1

            features = self.feature_extractor.compute_features(pyr_image, rect, new_container_rect)
            prediction = self.classifier.predict(features)
            if abs(1 - prediction) < 0.1:
                positives.append(self.upscale_int_tuple(rect, level))
        return positives

    def extract_features(self, img, rect_sets):
        img, scale_factor = self.length_resize(img, self.img_size)

        w, h = img.shape[1], img.shape[0]
        features = None

        for ind, rect_set in enumerate(rect_sets):
            container_rect, labeled_rects = rect_set

            # resize to match the starting size
            container_rect = tuple([int(round(i * scale_factor)) for i in container_rect])

            windows = self.get_windows(w, h, container_rect=container_rect)
            print 'extracting features from {} windows in container {}'.format(len(windows), ind)
            win_area = self.win_size ** 2

            positives = []
            negatives = []
            is_a_good_sample = {}

            # iterate through the set of windows for current container rect
            for window in windows:
                (x, y), level = window
                pyr_image = self.get_pyramid_image(img, level)
                win_rect = (x, y, self.win_size, self.win_size)

                # resize the container rect again based on the image pyramid level
                new_container_rect = self.downscale_int_tuple(container_rect, level)

                # determine the current window's label based on annotations
                # this part should be refactored into a separate function later
                for labeled_r in labeled_rects:
                    # resize to match the starting size, and then the pyramid level.
                    # todo: refactoring
                    new_labeled_r = tuple([int(round(i * scale_factor)) for i in labeled_r])
                    new_labeled_r = self.downscale_int_tuple(new_labeled_r, level)

                    lx, ly, lw, lh = new_labeled_r

                    i_rect = get_intersecting_rect2(win_rect, new_labeled_r)

                    # separated the if_blocks for readability
                    if i_rect is None:
                        negatives.append((win_rect, level))
                        continue

                    # intersection size
                    _, _, iw, ih = i_rect
                    i_area = iw * ih

                    ## note that this collects a tuple (window, pyramid level), not an upscaled window

                    # if the non-intersecting area is more than 20%, negative
                    non_intersecting_area = abs(win_area - i_area)
                    if non_intersecting_area > win_area * 0.12:
                        negatives.append((win_rect, level))
                        continue

                    # if the smaller side of the label is more than 20% different, negative
                    length_difference = abs(min(lw, lh) - self.win_size)
                    if length_difference > self.win_size * 0.12:
                        negatives.append((win_rect, level))
                        continue

                    # now positive
                    pos_sample = (win_rect, level)

                    # if the labeled area is nearly square, and this window overlaps most of it,
                    # then this is a good sample, and more like this should be added.
                    if abs(lw - lh) / float(max(lw, lh)) < 0.3 \
                            and non_intersecting_area < win_area * 0.08 \
                            and length_difference < win_area * 0.08:
                        is_a_good_sample[pos_sample] = True

                    positives.append(pos_sample)

            random.shuffle(negatives)
            negatives = negatives[:int(len(positives) * 1.3)]

            # self.plot_feature_windows(img, positives, negatives, is_a_good_sample, container_rect)

            # todo: reduce repetition
            for pos in positives:
                pos_rect, level = pos
                pyr_im = self.get_pyramid_image(img, level)
                pyr_container_rect = self.downscale_int_tuple(container_rect, level)
                feature_vector = self.feature_extractor.compute_features(pyr_im, pos_rect, pyr_container_rect)

                # right now I'm naively duplicating desired samples in the data set, but ideally I should add noise
                # and generate synthetic data from it.
                if pos in is_a_good_sample and is_a_good_sample[pos] is True:
                    count = 2
                else:
                    count = 1

                for i in range(count):
                    if features is None:
                        features = np.hstack((1, feature_vector)).reshape(1, feature_vector.shape[0])
                    else:
                        f_row = np.hstack((1, feature_vector)).reshape(1, feature_vector.shape[0])
                        np.vstack((feature_vector, f_row))
            for neg in negatives:
                neg_rect, level = neg
                pyr_im = self.get_pyramid_image(img, level)
                pyr_container_rect = self.downscale_int_tuple(container_rect, level)
                feature_vector = self.feature_extractor.compute_features(pyr_im, pos_rect, pyr_container_rect)
                if features is None:
                    features = np.hstack((0, feature_vector)).reshape(1, feature_vector.shape[0])
                else:
                    f_row = np.hstack((0, feature_vector)).reshape(1, feature_vector.shape[0])
                    np.vstack((feature_vector, f_row))

        return features

    def plot_feature_windows(self, img, pos, neg, is_good, container):
        rects = []
        colors = []
        for prect, l in neg:
            rect = self.upscale_int_tuple(prect, l)
            rects.append(rect)
            colors.append('blue')
        for prect, l in pos:
            rect = self.upscale_int_tuple(prect, l)
            rects.append(rect)
            if (prect, l) in is_good and is_good[(prect, l)] == True:
                colors.append('green')
            else:
                colors.append('magenta')
        rects.append(container)
        colors.append('yellow')
        plot_rects_on_image(img, rects, colors)

    def detect(self, img, container_rect):
        """
        :type img: ndarray
        :type container_rect: tuple[int]
        :rtype: list[tuple[int]]

        :param img: Image to detect positive response
        :param container_rect: Rectangular area (x, y, w, h) to run the detection
        :return: List of (x, y, w, h)
        """
        img, r = self.length_resize(img, self.img_size)
        container_rect = tuple([int(round(i * r)) for i in container_rect])

        w, h = img.shape[1], img.shape[0]
        windows = self.get_windows(w, h, container_rect=container_rect)

        return self.__detector(img, container_rect, windows)

        # split_windows = chunks(windows, self.n_processes)
        #
        # processes = []
        # for i in range(self.n_processes):
        #     p = Process(target=self.__detector_process, args=(img, container_rect, split_windows[i]))
        #     processes.append(p)
        #     p.start()
        #
        # for p in processes:
        #     p.join()
        #
        # # TODO: join and return
        # return

    def get_pyramid_image(self, img, level):
        """
        Get a downsampled version of the image from the image pyramid. The images are resized using cv2.resize()
        instead of cv2.PyrUp().
        """
        key = (id(img), level)
        if key in self.image_pyramid:
            return self.image_pyramid[key]
        else:
            size = img.shape[1], img.shape[0]
            dst_dist = self.downscale_int_tuple(size, level)
            resized_image = cv2.resize(img, dst_dist)
            self.image_pyramid[key] = resized_image
            return resized_image

    @staticmethod
    def length_resize(img, l):
        """
        Return (r_img, r) where r_img is a resized image such that the length of the
        longer side is equal to l, and the resize factor r is a number such that
        img_size * r = r_img_size.
        """
        imw, imh = img.shape[1], img.shape[0]
        if imw > imh:
            dst_size = (l, int(round(l * float(imh) / imw)))
            r = float(l) / imw
        else:
            dst_size = (int(l * float(imw) / imh), l)
            r = float(l) / imh
        r_img = cv2.resize(img, dst_size)
        return r_img, r

    def get_windows(self, im_w, im_h, skip=10, container_rect=None, include_size=None):
        """
        Get a list of window coordinates for further processing.

        :param im_w: Width of the source image.
        :param im_h: Height of the source image.
        :param skip: Number of pixels to skip.
        :param container_rect: (Optional) Tuple (x, y, w, h) of the containing border.
        :param include_size: (Optional) If True, return value will be a list of ((x, y, s, s), level)
        :return: List of ((x, y), level)
        """

        # Set the lower the upper limits
        if container_rect is None:
            xl, yl, xh, yh = 0, 0, im_w, im_h
        else:
            xl, yl, cw, ch = container_rect
            xh = min(xl + cw - self.win_size, im_w - self.win_size)
            yh = min(xl + ch - self.win_size, im_h - self.win_size)

        windows = []
        w, h = im_w, im_h
        xl_, xh_, yl_, yh_ = xl, xh, yl, yh
        level = 0
        _count = 0
        while min(w, h) > self.win_size:
            for x in range(xl, xh, skip):
                for y in range(yl, yh, skip):
                    if include_size is None:
                        window = (x, y)
                    else:
                        window = (x, y, self.win_size, self.win_size)
                    windows.append((window, level))

            level += 1
            w, h, xl, xh, yl, yh = self.downscale_int_tuple(
                (im_w, im_h, xl_, xh_ + self.win_size, yl_, yh_ + self.win_size), level)
            xh -= self.win_size
            yh -= self.win_size
        return windows

    def get_cropped_images(self, img, win_size=90, skip=2):
        """
        Get a list containing cropped images and the coordinates.

        :type img: ndarray
        :type win_size: int
        :type skip: int
        :rtype: list

        :param img: Source image
        :param win_size: Sliding window size
        :param skip: Number of pixels to skip when sliding
        :return: List of (cropped image, (x, y, w, h), downsize level)
        """
        i = 0
        win = []
        while min(img.shape[0:2]) > win_size:
            imw, imh = img.shape[1], img.shape[0]
            self.image_cache[i] = img
            for x in range(0, imw - win_size, skip):
                for y in range(0, imh - win_size, skip):
                    rect = (x, y, win_size, win_size)
                    cropped_img = crop_image(img, rect)
                    win.append((cropped_img, rect, i))
            new_size = (int(round(imw * self.resize_factor)), int(round(imh * self.resize_factor)))
            img = cv2.resize(img, new_size)
            i += 1
        return win

    def downscale_int_tuple(self, t, level=1):
        """
        Decrease the scale of the integer values in a given tuple.

        :type t: tuple
        :type level: int
        :rtype: tuple

        :param t: Tuple of integers.
        :param level: Number of levels to downsize. level=0 is no change.
        :return: t * r ^ levels for r < 1
        """
        return tuple([int(round(item * (self.resize_factor ** level))) for item in t])

    def upscale_int_tuple(self, t, level=1):
        """
        Increase the scale of the integer values in a given tuple.

        :type t: tuple
        :type level: int
        :rtype: tuple

        :param t: Tuple of integers.
        :param level: Number of levels to upsize. level=0 is no change.
        :return: t / r ^ levels for r < 1
        """
        return tuple([int(round(item / (self.resize_factor ** level))) for item in t])
