from multiprocessing import Pool, Process
import cv2
import itertools
from helpers.image_operation_helpers import crop_image
from helpers.parallelization_helpers import chunks

__author__ = 'Daeyun Shin'


class SlidingWindowDetector:
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
        positives = []
        for window in windows:
            (x, y), level = window
            pyr_image = self.get_pyramid_image(img, level)
            rect = (x, y, self.win_size, self.win_size)
            features = self.feature_extractor.compute_features(pyr_image, rect, container_rect)
            prediction = self.classifier.predict(features)
            if abs(1 - prediction) < 0.1:
                positives.append(self.upscale_int_tuple(rect, level))
        return positives

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
        container_rect = tuple([int(round(i*r)) for i in container_rect])

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
            size = img.shape[1], img.shape[1]
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
            r = float(l)/imw
        else:
            dst_size = (int(l * float(imw) / imh), l)
            r = float(l)/imh
        r_img = cv2.resize(img, dst_size)
        return r_img, r

    def get_windows(self, im_w, im_h, skip=2, container_rect=None, include_size=None):
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
        level = 0
        while min(w, h) > self.win_size:
            for x in range(xl, xh, skip):
                for y in range(yl, yh, skip):
                    if include_size is None:
                        window = (x, y)
                    else:
                        window = (x, y, self.win_size, self.win_size)
                    windows.append((window, level))
            level += 1
            w, h = self.downscale_int_tuple((im_w, im_h), level)
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
