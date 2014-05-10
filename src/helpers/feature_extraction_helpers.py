import cv2
import numpy as num
from skimage.feature import hog

__author__ = 'Daeyun Shin'


def compute_hog(img):
    """
    @param img: OpenCV image
    @returns 1x128 tuple
    @type img: ndarray
    @rtype: tuple
    """
    w, h = img.shape[1], img.shape[0]
    assert h == w
    if h != 96:
        img = cv2.resize(img, (96, 96))

    #  cv2.imread() loads images as BGR while numpy.imread() loads them as RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to scikit-image's image type
    img = img.astype(num.float32)/255.0
    fd = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(2, 2))
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
        assert (mask.dtype == num.uint8)
        assert (mask.shape == img.shape)

    hist = cv2.calcHist(images, channels, mask, hist_size, ranges)

    # Normalized so that their entries sum up to 1
    hist = cv2.normalize(hist, norm_type=cv2.NORM_L1)

    return hist


def chi_squared_distance(hist1, hist2):
    """
    Compute d(x,y) = sum((xi-yi)^2/(xi+yi)) / 2 between two histograms.
    hist1 and hist2 must be normalized so that their entries sum up to 1
    @type hist1: ndarray
    @type hist2: ndarray
    @rtype: np.float32
    """
    x = hist1.flatten()
    y = hist2.flatten()
    return num.sum(num.square(x-y)/(x+y))/2