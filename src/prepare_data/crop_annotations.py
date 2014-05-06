import os
from helpers.config_helpers import parse_annotations
from helpers.io_helpers import search_files_by_extension, path_to_filename, make_sure_dir_exists
import cv2
from helpers.image_processing_helpers import rotate_image, get_ROIs


def extract_labeled_images(input_image_dir, input_annotation_dir, out_dir, callback):
    """
    @type input_image_dir: string
    @type input_annotation_dir: string
    @type out_dir: string
    """

    annotations = parse_annotations(input_annotation_dir)
    image_paths = search_files_by_extension(input_image_dir, ["JPG"])
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_filename = path_to_filename(image_path)
        annotation = annotations[image_filename]
        image_info = annotation['image info']

        (img_w, img_h) = image.shape[1], image.shape[0]
        if (img_w, img_h) != (image_info['w'], image_info['h']):
            image = cv2.resize(image, (image_info['w'], image_info['h']))

        orientation = image_info['orientation']
        if orientation % 360 != 0:
            image = rotate_image(image, -orientation)

        for label, rects in annotation['rects'].items():
            # Translate origin to the top left corner
            adjusted_rects = []
            # Get rotated image size
            (img_w, img_h) = image.shape[1], image.shape[0]
            for rect in rects:
                x, y, w, h = rect
                new_rect = x+img_w/2, y+img_h/2, w, h
                adjusted_rects.append(map(int, new_rect))

            # Image patches annotated with the current label
            ROIs = get_ROIs(image, adjusted_rects)

            out_dir_label = os.path.join(out_dir, label)
            make_sure_dir_exists(out_dir)

            for index, img_patch in enumerate(ROIs):
                x, y, w, h = adjusted_rects[index]
                patch_info = {
                    'patch': {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'label': label,
                    },
                    'source': {
                        'orientation': orientation,
                        'path': image_path,
                    },
                    'out dir': out_dir_label
                }

                callback(img_patch, patch_info)
