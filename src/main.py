import argparse
import json
import os
from pprint import pprint
import textwrap
import cv2
import numpy as np
from helpers.config_helpers import *
from helpers.io_helpers import *
from helpers.image_processing_helpers import *


def main():
    parser = argparse.ArgumentParser(description='CS 543 Project')
    parser.add_argument('--config-file', nargs='?', default='../config.yaml',
                        help='path to config.yaml (default: ../config.yaml)')
    args = parser.parse_args()
    config_path = args.config_file

    try:
        config = parse_config(get_absolute_path(config_path))
    except Exception as e:
        pretty_print_exception("Could not load {}".format(config_path), e)
        return

    annotations = parse_annotations(config['paths']['input']['annotation'])

    image_paths = search_files_by_extension(config['paths']['input']['image'], ["JPG"])
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_filename = path_to_filename(image_path)
        annotation = annotations[image_filename]
        image_info = annotation['image info']

        (img_w, img_h) = image.shape[1], image.shape[0]
        if (img_w, img_h) != (image_info['w'], image_info['h']):
            image = cv2.resize(image, (image_info['w'], image_info['h']))

        if image_info['orientation'] % 360 != 0:
            image = rotate_image(image, -image_info['orientation'])

        for label, rects in annotation['rects'].items():
            ## Move origin to the top left corner
            adjusted_rects = []
            # Get rotated image size
            (img_w, img_h) = image.shape[1], image.shape[0]
            for rect in rects:
                x, y, w, h = rect
                new_rect = x+img_w/2, y+img_h/2, w, h
                adjusted_rects.append(map(int, new_rect))

            # Image patches annotated with the current label
            ROIs = get_ROIs(image, adjusted_rects)

            out_dir = os.path.join(config['paths']['output']['image patch'], label)
            make_sure_dir_exists(out_dir)

            for index, img_patch in enumerate(ROIs):
                name, ext = image_filename.rsplit('.', 1)
                x, y, w, h = adjusted_rects[index]
                out_filename = os.path.join(out_dir, "{name}.{x}.{y}.{w}.{h}.{o}.{ext}".format(**{
                    'name': name,
                    'ext': ext,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'o': image_info['orientation'],
                }))
                cv2.imwrite(out_filename, img_patch)


if __name__ == '__main__':
    main()