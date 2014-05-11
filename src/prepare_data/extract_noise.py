import cv2
import numpy as np
import hashlib
from numpy.core.multiarray import ndarray, array
from helpers.io_helpers import search_files_by_extension, path_to_filename
from helpers.config_helpers import parse_annotations
from helpers.geometry_helpers import rect_to_polygon, rotate_rects, find_overlapping_polygon_area
from helpers.image_operation_helpers import rotate_image, crop_images
from helpers.plotting_helpers import plot_polygons_on_image

__author__ = 'Daeyun Shin'


def extract_noise(input_image_dir, input_annotation_dir, out_dir, callback, max_width=96, instance_id=None, num_instances=None):
    """
    @type input_image_dir: string
    @type input_annotation_dir: string
    @type out_dir: string
    @type callback: function
    """

    annotations = parse_annotations(input_annotation_dir)
    image_paths = search_files_by_extension(input_image_dir, ["JPG"])
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_filename = path_to_filename(image_path)

        # Distributed processing
        if instance_id is not None and num_instances is not None:
            if (int(hashlib.md5(image_path).hexdigest(), 16) % num_instances) != instance_id:
                continue
            print "instance id: {}. total number of instances {}".format(instance_id, num_instances)
            print "processing file {}".format(image_path)

        try:
            annotation = annotations[image_filename]
            image_info = annotation['image info']
        except:
            print "annotation is not available for {}".format(image_filename)
            continue

        (img_w, img_h) = image.shape[1], image.shape[0]
        if (img_w, img_h) != (image_info['w'], image_info['h']):
            image = cv2.resize(image, (image_info['w'], image_info['h']))

        theta = image_info['orientation']
        dthetas = [0, 15]
        for dtheta in dthetas:
            orientation = -theta + dtheta
            rotated_image = rotate_image(image, orientation)

            print theta, orientation

            positive_rects = []
            border_rects = None
            image_rect = [-image.shape[1]/2, -image.shape[0]/2, image.shape[1], image.shape[0]]
            for k, v in annotation['rects'].items():
                if k == 'border':
                    border_rects = v
                else:
                    positive_rects += v

            if border_rects is None:
                print 'border rectangle must exist.'
                continue

            # index 0: original image, 1 to k: product border, k+1 to n-1: positive samples
            rects = np.array([image_rect] + border_rects + positive_rects)
            # change the origin to top left and handle rotations
            rects[:, 0] += rotated_image.shape[1]/2.0
            rects[:, 1] += rotated_image.shape[0]/2.0
            poly_rects = rect_to_polygon(rects)
            rotation_center = (rotated_image.shape[1]/2.0, rotated_image.shape[0]/2.0)
            rotated_poly_rects = rotate_rects(poly_rects, rotation_center, dtheta)

            p_image = rotated_poly_rects[0]
            p_borders = rotated_poly_rects[1:len(border_rects)+1]
            p_positives = rotated_poly_rects[len(border_rects)+1:]

            # plot_polygons_on_image(rotated_image, [image_poly, border_poly, positive_poly], ['red', 'blue', 'yellow'])

            r_img_w = rotated_image.shape[1]
            r_img_h = rotated_image.shape[0]

            min_size = 80
            max_size = min(r_img_w, r_img_h)
            window_size_growth = 1.6

            size = min_size
            while size <= max_size:
            	skip_size = max(size/2, 50)
                for x in range(0, r_img_w - size, skip_size):
                    for y in range(0, r_img_h - size, skip_size):
                        p_window = rect_to_polygon(array([(x, y, size, size)]))[0]
                        window_area = size**2 - 1  # subtract 1 in case of floating point errors
                        win_img_overlap = find_overlapping_polygon_area(p_window, p_image)
                        if win_img_overlap is None or win_img_overlap < window_area:
                            continue
                        skip = False
                        for p_border in p_borders:
                            win_border_overlap = find_overlapping_polygon_area(p_window, p_border)
                            if win_img_overlap is None or win_border_overlap < window_area*0.75:
                                skip = True
                                break
                        if skip:
                            continue
                        for p_positive in p_positives:
                            win_pos_overlap = find_overlapping_polygon_area(p_window, p_positive)
                            if win_pos_overlap is not None and win_pos_overlap > window_area*0.55:
                                skip = True
                                break
                        if skip:
                            continue

                        patch_info = {
                            'patch': {
                                'x': x,
                                'y': y,
                                'w': size,
                                'h': size,
                                'label': 'negative',
                                },
                            'source': {
                                'theta': theta,  # orientation
                                'dtheta': dtheta,
                                'path': image_path,
                                },
                            'out dir': out_dir
                        }

                        img_patch = crop_images(rotated_image, [(x, y, size-1, size-1)])[0]

                        if size > max_width:
                            img_patch = cv2.resize(img_patch, (max_width, max_width))

                        callback(img_patch, patch_info)

                size = int(size * window_size_growth)
