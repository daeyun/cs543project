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
from helpers.io_helpers import get_absolute_path, search_files_by_extension, pretty_print_exception, \
    make_sure_dir_exists
from helpers.parallelization_helpers import chunks
from helpers.plotting_helpers import plot_polygons_on_image
from multiprocessing import Process

__author__ = 'Daeyun Shin'


def shuffle_in_unison_inplace(a, b):
    # http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    p = num.random.permutation(len(a))
    return a[p], b[p]


def info(msg):
    print 'process {}: {}'.format(os.getpid(), msg)


def feature_extractor_process(X, Y, annotations, source_img_dir, out_dir, process_id, instance_id):
    assert len(X) == len(Y)
    info('starting. received {} items.'.format(len(X)))

    source_img_dict = {}

    make_sure_dir_exists(out_dir)

    for idx, filename in enumerate(X):
        img = cv2.imread(filename)

        # HOG feature vector
        fd = compute_hog(img)

        source_filename, x, y, w, h, theta, dtheta = unpack_filename(filename)
        source_file_path = os.path.join(source_img_dir, source_filename)

        if source_file_path not in source_img_dict:
            source_img = cv2.imread(source_file_path)
            source_img_dict[source_file_path] = source_img
        else:
            source_img = source_img_dict[source_file_path]

        orientation = -theta + dtheta
        source_img = rotate_image(source_img, orientation)
        patch_rect = (x, y, w, h)

        try:
            border_rects = num.array(annotations[source_filename]['rects']['border'])
        except Exception as e:
            pretty_print_exception('Annotation is not available for {}'.format(filename), e)
            continue

        border_rects[:, 0] += source_img.shape[1] / 2.0
        border_rects[:, 1] += source_img.shape[0] / 2.0
        border_rect = None

        for r in border_rects:
            patch_poly = rect_to_polygon(num.array([patch_rect]))
            border_poly = rect_to_polygon(num.array([r]))
            area = find_overlapping_polygon_area(patch_poly[0], border_poly[0])
            if area is not None and area > w * h * 0.5:
                border_rect = r

        if border_rect is None:
            print "Border not found"
            continue

        color_contrasts = compute_surrounding_color_contrast(source_img, patch_rect, border_rect)
        cc_dissimilarity = [1 / (i / 10 + 1) for i in color_contrasts]

        # normalize x, y, w, h values to be used as features.
        # TODO: vectorize this
        cx = x + w / 2.0
        cy = y + h / 2.0
        bx, by, bw, bh = border_rect
        b_cx = bx + bw / 2.0
        b_cy = by + bh / 2.0
        dcx = cx - b_cx
        dcy = cy - b_cy

        dcx = dcx / bw + 0.5
        dcy = dcy / bh + 0.5
        dw = w / bw
        dh = h / bh

        feature_vector = hstack((fd, cc_dissimilarity, num.array([dcx, dcy, dw, dh])))

        out_filename = "{instance_id}_{process_id}_{y}.txt".format(instance_id=instance_id, process_id=process_id, y=Y[idx])
        out_path = os.path.join(out_dir, out_filename)
        with open(out_path, "a") as myfile:
            myfile.write(str(Y[idx]) + ', ' + ', '.join(map(str, feature_vector)) + '\n')

        info('saved {} as {}'.format(filename, out_path))
        # info(feature_vector)

    info('ending')


def extract_features(source_img_dir, annotation_dir, pos_set_dir, neg_set_dir, out_dir, instance_id=None,
                     num_instances=None, num_processes=None):

    annotations = parse_annotations(annotation_dir)
    positive_set_paths = search_files_by_extension(pos_set_dir, ["JPG", "PNG"], is_recursive=True)
    negative_set_paths = search_files_by_extension(neg_set_dir, ["JPG", "PNG"])

    X = num.array(positive_set_paths + negative_set_paths)
    Y = num.array([1] * len(positive_set_paths) + [0] * len(negative_set_paths))

    num.random.seed(383838)
    X, Y = shuffle_in_unison_inplace(X, Y)
    print 'Shuffled with seed value 0. X hash: {}, Y hash: {}'.format(hash(tuple(X)), hash(tuple(Y)))

    assert len(X) == len(Y)
    n_total_items = len(Y)

    X = chunks(X, num_instances)[instance_id]
    Y = chunks(Y, num_instances)[instance_id]

    Xs = chunks(X, num_processes)
    Ys = chunks(Y, num_processes)

    assert len(Xs) == len(Ys)

    processes = [0] * num_processes
    for i in range(num_processes):
        processes[i] = Process(target=feature_extractor_process, args=(Xs[i], Ys[i], annotations, source_img_dir, out_dir, i, instance_id))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    info('Exiting')

