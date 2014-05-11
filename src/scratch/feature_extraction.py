import os
import cv2
import matplotlib.pyplot as plt
from numpy.ma import hstack
from skimage import data, img_as_ubyte, color
from skimage.exposure import exposure
from skimage.feature import hog
import numpy as num
from helpers.config_helpers import parse_config, parse_annotations, unpack_filename
from helpers.feature_extraction_helpers import compute_hog, compute_lab_histogram, compute_surrounding_color_contrast
from helpers.geometry_helpers import find_overlapping_polygon_area, rect_to_polygon
from helpers.image_operation_helpers import rotate_image
from helpers.io_helpers import get_absolute_path, search_files_by_extension
from helpers.plotting_helpers import plot_polygons_on_image

config_path = '../../config.yaml'

config = parse_config(get_absolute_path(config_path))
# g_config is a global variable
global g_config
g_config = config

source_img_dir = g_config['paths']['input']['initial']['image']
annotation_dir = g_config['paths']['input']['initial']['annotation']

positive_set_dir = g_config['paths']['input']['step one']['positive']
negative_set_dir = g_config['paths']['input']['step one']['negative']

annotations = parse_annotations(annotation_dir)
positive_set_paths = search_files_by_extension(positive_set_dir, ["JPG", "PNG"], is_recursive=True)
negative_set_paths = search_files_by_extension(negative_set_dir, ["JPG", "PNG"])

for idx, filename in enumerate(positive_set_paths + negative_set_paths):
    if idx < len(positive_set_paths):
        label = 1
    else:
        label = 0

    img = cv2.imread(filename)
    fd = compute_hog(img)
    # print fd

    source_filename, x, y, w, h, theta, dtheta = unpack_filename(filename)
    source_file_path = os.path.join(source_img_dir, source_filename)

    source_img = cv2.imread(source_file_path)
    orientation = -theta + dtheta
    source_img = rotate_image(source_img, orientation)

    patch_rect = (x, y, w, h)

    border_rects = num.array(annotations[source_filename]['rects']['border'])
    border_rects[:, 0] += source_img.shape[1]/2.0
    border_rects[:, 1] += source_img.shape[0]/2.0
    border_rect = None

    for r in border_rects:
        patch_poly = rect_to_polygon(num.array([patch_rect]))
        border_poly = rect_to_polygon(num.array([r]))
        area = find_overlapping_polygon_area(patch_poly[0], border_poly[0])
        if area is not None and area > w*h*0.5:
            border_rect = r

    if border_rect is None:
        print "Border not found"
        continue

    color_contrasts = compute_surrounding_color_contrast(source_img, patch_rect, border_rect)
    cc_dissimilarity = [1/(i/10+1) for i in color_contrasts]
    print fd
    print color_contrasts
    print cc_dissimilarity

    # normalize x, y, w, h values to be used as features.
    # TODO: vectorize this
    cx = x + w/2.0
    cy = y + h/2.0
    bx, by, bw, bh = border_rect
    b_cx = bx + bw/2.0
    b_cy = by + bh/2.0
    dcx = cx - b_cx
    dcy = cy - b_cy

    dcx = dcx/bw + 0.5
    dcy = dcy/bh + 0.5
    dw = w/bw
    dh = h/bh

    feature_vector = hstack((fd, cc_dissimilarity, num.array([dcx, dcy, dw, dh])))

    print feature_vector