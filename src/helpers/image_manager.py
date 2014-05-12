import cv2
from helpers.config_helpers import parse_annotations
from helpers.feature_extraction_helpers import get_intersecting_rect2
from helpers.image_operation_helpers import rotate_image
from helpers.io_helpers import path_to_filename
from pprint import pprint

__author__ = 'Daeyun Shin'

class ImageManager:
    def __init__(self, annotation_dir):
        self.annotations = parse_annotations(annotation_dir)
        self.cache = {}

    def load_image(self, path, is_cached=True):
        if is_cached and path in self.cache:
            return self.cache[path]

        image = cv2.imread(path)
        img_w, img_h = image.shape[1], image.shape[0]
        filename = path_to_filename(path)

        if filename not in self.annotations:
            raise Exception("Annotation file not found for {}".format(path))
            return

        image_info = self.annotations[filename]['image info']
        aw = image_info['w']
        ah = image_info['h']
        if (img_w, img_h) != (aw, ah):
            image = cv2.resize(image, (aw, ah))

        container_rect = self.annotations[filename]['rects']['border'][0]
        theta = image_info['orientation']
        rotated_image = rotate_image(image, -theta)

        container_rect[0] += rotated_image.shape[1]/2.0
        container_rect[1] += rotated_image.shape[0]/2.0

        return_value = image, container_rect
        self.cache[path] = return_value
        return return_value

    def load_annotated_image(self, path, is_cached=True):
        if is_cached and path in self.cache:
            return self.cache[path]

        image = cv2.imread(path)
        img_w, img_h = image.shape[1], image.shape[0]
        filename = path_to_filename(path)

        if filename not in self.annotations:
            raise Exception("Annotation file not found for {}".format(path))
            return None

        image_info = self.annotations[filename]['image info']
        aw = image_info['w']
        ah = image_info['h']
        if (img_w, img_h) != (aw, ah):
            image = cv2.resize(image, (aw, ah))

        container_rects = self.annotations[filename]['rects']['border']
        theta = image_info['orientation']
        rotated_image = rotate_image(image, -theta)

        labeled_rects = []
        if 'illustration' in self.annotations[filename]['rects']:
            labeled_rects += self.annotations[filename]['rects']['illustration']
        if 'logo' in self.annotations[filename]['rects']:
            labeled_rects += self.annotations[filename]['rects']['logo']

        # todo: vectorize this
        for rect in container_rects+labeled_rects:
            rect[0] += rotated_image.shape[1]/2.0
            rect[1] += rotated_image.shape[0]/2.0

        label_rect_sets = []
        for container_rect in container_rects:
            label_rect_set = []
            for rect in labeled_rects:
                x, y, w, h = rect
                r_area = w*h
                i_rect = get_intersecting_rect2(container_rect, rect)

                if i_rect is None:
                    continue

                ix, iy, iw, ih = i_rect
                i_area = iw*ih

                # non intersecting area is more than half
                if float(abs(r_area-i_area)) > r_area*0.5:
                    continue

                label_rect_set.append(rect)
            label_rect_sets.append((container_rect, label_rect_set))

        return_value = image, label_rect_sets
        self.cache[path] = return_value
        return return_value
