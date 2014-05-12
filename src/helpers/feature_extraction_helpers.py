import cv2
import numpy as num
from skimage.feature import hog
import time
from helpers.geometry_helpers import rect_to_polygon, find_overlapping_polygon, expand_rect, poly_to_rect
from helpers.image_operation_helpers import crop_image

__author__ = 'Daeyun Shin'


def cv_hog(img):
    """
    This is faster than scikit-image's HOG
    http://docs.opencv.org/trunk/doc/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
    """
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = num.int32(bin_n * ang / (2 * num.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [num.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = num.hstack(hists)
    hist = cv2.normalize(hist, norm_type=cv2.NORM_L1)
    return hist


def compute_hog(img):
    """
    @param img: OpenCV image
    @returns 1x128 tuple
    @type img: ndarray
    @rtype: tuple
    """
    w, h = img.shape[1], img.shape[0]
    assert h == w
    if h != 60:
        img = cv2.resize(img, (60, 60))

    #  cv2.imread() loads images as BGR while numpy.imread() loads them as RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # decided not to used scikit-image's hog
    # convert to scikit-image's image type
    # img = img.astype(num.float32) / 255.0
    # fd = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(2, 2))

    fd = cv_hog(img)
    return fd

def compute_lab_histogram(img, mask=None):
    """
    Compute histogram in 4x8x8 LAB colorspace
    @param img: OpenCV image. BGR colorspace.
                img should be CV_8U (img.astype('uint8')
    @param mask: If the matrix is not empty, it must be an 8-bit array of the same size as img.
                 The non-zero mask elements mark the array elements counted in the histogram.
                 The mask argument must be 8-bit (mask.astype('uint8')).
    @type img: ndarray
    @type mask: ndarray
    """
    # Convert to LAB space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Parameters for calcHist
    images = [lab_img]
    channels = [0, 1, 2]
    hist_size = [4, 8, 8]

    # Although the documentation suggests it should be an array of array, it should in fact be a flat list.
    ranges = [0, 255, 0, 255, 0, 255]

    if mask is not None:
        mask.astype(num.uint8)
        assert (mask.dtype == num.uint8)
        assert (mask.shape[0] == img.shape[0])
        assert (mask.shape[1] == img.shape[1])

    hist = cv2.calcHist(images, channels, mask, hist_size, ranges)

    # Normalized so that their entries sum up to 1
    hist = cv2.normalize(hist, norm_type=cv2.NORM_L1)

    return hist


def get_mask(size, larger_rect, smaller_rect):
    """
    larger_rect is a rect tuple (x, y, w, h) that contains smaller_rect.
    @type larger_rect: ndarray
    @type smaller_rect: ndarray
    @rtype ndarray
    """
    w, h = size
    lx, ly, lw, lh = larger_rect
    sx, sy, sw, sh = smaller_rect

    mask = num.ones((h, w), dtype=num.uint8)
    mask[sx - lx:sx - lx + lw - 1, sx - lx:sy - ly + lh - 1] = 0

    return mask


def get_intersecting_rect(rect1, rect2, is_integer=True):
    """
    Given two (x, y, w, h) rects, find the intersecting (x2, y2, w2, h2) rect
    @type rect1: tuple
    @type rect2: tuple
    @type is_integer: bool
    """
    rect1_poly = rect_to_polygon([rect1])[0]
    rect2_poly = rect_to_polygon([rect2])[0]
    overlap_poly = find_overlapping_polygon(rect1_poly, rect2_poly)
    overlap_rect = poly_to_rect(overlap_poly)

    if is_integer:
        return tuple([int(i) for i in overlap_rect])
    else:
        overlap_rect


def get_intersecting_rect2(rect1, rect2):
    """
    faster
    """
    l1, t1, w1, h1 = rect1
    l2, t2, w2, h2 = rect2
    r1 = l1 + w1
    r2 = l2 + w2
    b1 = t1 + h1
    b2 = t2 + h2
    x = max(l1, l2)
    y = max(t1, t2)
    w = min(r1, r2) - x
    h = min(b1, b2) - y
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def compute_surrounding_color_contrast(img, rect, border_rect, layers=2, background_hist=None):
    """
    Measure of the dissimilarity of a window to its immediate surrounding area.
    Reference: "Measuring the objectness of image windows" Alexe et al.
    @type rect: tuple
    @type border_rect: tuple
    """
    chi_squared_distances = [0] * layers

    rect_img = crop_image(img, rect)
    window_hist = compute_lab_histogram(rect_img)

    # rects = [rect]

    prev_rect = rect
    for level in range(layers):
        expanded_rect = expand_rect(prev_rect, radius_increase=1.5)
        expanded_rect = get_intersecting_rect(expanded_rect, border_rect)
        # rects.append(expanded_rect)
        expanded_rect_img = crop_image(img, expanded_rect)
        mask = get_mask(expanded_rect, prev_rect)

        surrounding_hist = compute_lab_histogram(expanded_rect_img, mask=mask)

        # method=1 is the same as cv.CV_COMP_CHISQR
        chi_squared_distances[level] = cv2.compareHist(window_hist, surrounding_hist, method=1)

        prev_rect = expanded_rect

    # # Uncomment to plot the surrounding areas
    # rects_list = [rect_to_polygon([border_rect])[0]]
    # for r in rects:
    #     rects_list.append(rect_to_polygon([r])[0])
    # plot_polygons_on_image(img, rects_list, ['red', 'blue', 'yellow', 'green'], will_block=True)

    if background_hist is None:
        rect_img = crop_image(img, border_rect)
        background_hist = compute_lab_histogram(rect_img)

    background_hist_dist = cv2.compareHist(window_hist, background_hist, method=1)
    chi_squared_distances.append(background_hist_dist)

    return chi_squared_distances
