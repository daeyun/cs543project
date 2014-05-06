import cv2
import numpy as np
from numpy.ma import empty_like


def rotate_image(image, angle=90):
    """
    Rotate the given image without cropping
    @type image: ndarray
    @type angle: int
    """
    center = tuple(np.multiply(image.shape[:2][::-1], 0.5))

    rot_mat = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0), [0, 0, 1]])

    rotation_mat = np.matrix(rot_mat[0:2, 0:2])

    rotated_corners = np.matrix(
        [[-center[0], center[1]],
         [center[0], center[1]],
         [-center[0], -center[1]],
         [center[0], -center[1]]]
    ) * rotation_mat

    x = np.reshape(np.array(rotated_corners[:, 0]), 4)
    y = np.reshape(np.array(rotated_corners[:, 1]), 4)

    right_bound = max(x)
    left_bound = min(x)
    top_bound = max(y)
    bot_bound = min(y)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    dx = (new_w * 0.5) - center[0]
    dy = (new_h * 0.5) - center[1]

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    return result


def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def get_ROIs(image, rects):
    """
    @type image: ndarray
    @type rects: list
    @type orientation: int
    """
    results = []
    for rect in rects:
        x, y, w, h = rect
        roi = image[y:y + h + 1, x:x + w + 1]
        results.append(roi)

    return results


def find_area(array):
    """
    >>> find_area([(0, 0), (10, 0), (10, 10), (0, 10),  (0, 0)])
    100

    @type array: list
    @rtype: float
    Reference: http://www.arachnoid.com/area_irregular_polygon/index.html
    """
    a = 0
    ox, oy = array[0]
    for x, y in array[1:]:
        a += (x * oy - y * ox)
        ox, oy = x, y
    return abs(a) / 2


def line_intersection(line1, line2):
    """
    >>> x, y = line_intersection([(0,0),(10,10)], [(10,0),(0,10)])
    >>> round(x, 2), round(y, 2)
    (5.0, 5.0)

    >>> str(line_intersection([(10,10),(10,11)], [(-10,-10),(-19,-10)]))
    'None'
    """
    a = np.linalg.det(line1)
    b = np.linalg.det(line2)
    c = line1[0][0] - line1[1][0]
    d = line2[0][0] - line2[1][0]
    e = line1[0][1] - line1[1][1]
    f = line2[0][1] - line2[1][1]
    g = np.linalg.det([[a, c],[b, d]])
    h = np.linalg.det([[c, e],[d, f]])
    i = np.linalg.det([[a, e],[b, f]])

    x = g/h
    y = i/h
    lx = [line1[0][0], line1[1][0]]
    ly = [line1[0][1], line1[1][1]]
    lx.sort()
    ly.sort()

    if lx[0] <= x <= lx[1] and ly[0] <= y <= ly[1]:
        return x, y

    return None



def rotate_rects(rects, center, angle):
    """
    @param rects: array of (x, y, w, h)
    @returns: array of [(x1, y1), (x2, y2), (x3, y3), (x4, y4))]
    @type rects: list
    @type center: tuple
    @type angle: float
    @rtype: list
    """
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # TODO


if __name__ == "__main__":
    import doctest

    doctest.testmod()
