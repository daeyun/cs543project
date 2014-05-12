import cv2
from helpers.image_operation_helpers import crop_image

__author__ = 'Daeyun Shin'

"""
multiscale, square sliding window
"""


class SlidingWindowClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        self.image_cache = {}
        self.resize_factor = 0.8
        self.positive_rois = []

    def classifier_thread(self, rois):
        for roi in rois:
            img, rect, level = roi
            self.extract_features(img, rect, level)
            self.classifier.predict(img)

    def classify(self, filename='../scratch/img/DSCF0876.JPG'):
        img = cv2.imread(filename)
        self.img = self.max_length_resize(img, 700)
        self.get_cropped_images(self.img)

    def max_length_resize(self, img, max_length):
        imw, imh = img.shape[1], img.shape[0]
        if max(imw, imh) > max_length:
            if imw > imh:
                dst_size = (max_length, max_length * imh / imw)
            else:
                dst_size = (max_length * imw / imh, max_length)
            img = cv2.resize(img, dst_size)
        return img

    def get_cropped_images(self, img, win_size=90, skip=2):
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

    def restore_roi(self, rect, level):
        win_rect = tuple([int(round(s / (self.resize_factor ** level))) for s in rect])
