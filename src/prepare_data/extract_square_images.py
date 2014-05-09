import os
import cv2
from numpy import ndarray
from helpers.config_helpers import parse_annotations
from helpers.geometry_helpers import rect_to_polygon, find_overlapping_polygon_area
from helpers.image_operation_helpers import get_ROIs, rotate_image
from helpers.io_helpers import search_files_by_extension, path_to_filename

__author__ = 'Daeyun Shin'


def extract_square_images(input_image_dir, input_annotation_dir, out_dir, callback, max_side=256):
    annotations = parse_annotations(input_annotation_dir)
    image_paths = search_files_by_extension(input_image_dir, ["JPG"])
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_filename = path_to_filename(image_path)
        try:
            annotation = annotations[image_filename]
        except:
            print "annotation is not available for {}".format(image_filename)
            continue

        image_info = annotation['image info']

        (img_w, img_h) = image.shape[1], image.shape[0]
        if (img_w, img_h) != (image_info['w'], image_info['h']):
            image = cv2.resize(image, (image_info['w'], image_info['h']))

        orientation = image_info['orientation']
        if orientation % 360 != 0:
            image = rotate_image(image, -orientation)

        for label, rects in annotation['rects'].items():
            if label == 'border':
                continue

            # Translate origin to the top left corner
            adjusted_rects = []
            # Get rotated image size
            (img_w, img_h) = image.shape[1], image.shape[0]
            for rect in rects:
                x, y, w, h = rect
                new_rect = x+img_w/2, y+img_h/2, w, h
                adjusted_rects.append(map(int, new_rect))

            squares = []
            for rect in adjusted_rects:
                x, y, w, h = rect
                if w > h:
                    end = w-h+1
                    step = max(min(end/3, h/2), 1)
                    for dx in range(0, end, step):
                        squares.append((x+dx, y, h, h))
                else:
                    end = h-w+1
                    step = max(min(end/3, h/2), 1)
                    for dy in range(0, end, step):
                        squares.append((x, y+dy, w, w))

            # Image patches annotated with the current label
            ROIs = get_ROIs(image, squares)

            out_dir_label = os.path.join(out_dir, label)

            for index, img_patch in enumerate(ROIs):
                x, y, w, h = squares[index]

                patch_info = {
                    'patch': {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'label': label,
                        },
                    'source': {
                        'theta': orientation,
                        'path': image_path,
                        },
                    'out dir': out_dir_label
                }

                assert(w == h)

                if w > max_side:
                    img_patch = cv2.resize(img_patch, (max_side, max_side))

                callback(img_patch, patch_info)