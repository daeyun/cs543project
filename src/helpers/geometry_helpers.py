import cv2
import math
import numpy as np
import matplotlib.path as mpl_path
from numpy.ma import vstack, ones

__author__ = 'Daeyun Shin'


def find_polygon_area(array):
    """
    >>> find_polygon_area([(0, 0), (10, 0), (10, 10), (0, 10),  (0, 0)])
    100

    @type array: list
    @rtype: float
    Reference: http://www.arachnoid.com/area_irregular_polygon/index.html
    """
    if type(array) is np.matrix:
        array = map(tuple, array.A)
    elif type(array) is np.ndarray:
        array = map(tuple, array)

    a = 0
    ox, oy = array[0]
    for x, y in array[1:]:
        a += (x * oy - y * ox)
        ox, oy = x, y
    return abs(a) / 2


def line_intersection(line1, line2, places=9):
    """
    >>> line_intersection([(0,0),(10,10)], [(10,0),(0,10)])
    (5.0, 5.0)
    >>> line_intersection([(0,0),(0,10)], [(5,0),(5,10)])
    >>> line_intersection([(0,0),(10,10)], [(5,5),(5,2)])
    (5.0, 5.0)
    >>> line_intersection([(0,0),(10,0)], [(5,5),(5,-1)])
    (5.0, -0.0)
    """
    a = np.linalg.det(line1)
    b = np.linalg.det(line2)
    c = line1[0][0] - line1[1][0]
    d = line2[0][0] - line2[1][0]
    e = line1[0][1] - line1[1][1]
    f = line2[0][1] - line2[1][1]
    g = np.linalg.det([[a, c], [b, d]])
    h = np.linalg.det([[c, e], [d, f]])
    i = np.linalg.det([[a, e], [b, f]])

    if h == 0:
        # lines are parallel
        return None

    x, y = g / h, i / h

    if places is None:
        return x, y
    return round(x, places), round(y, places)


def line_segment_intersection(line1, line2):
    """
    >>> line_segment_intersection([(0,0),(10,0)], [(5,5),(5,2)])
    >>> line_segment_intersection([(0,0),(10,0)], [(5,5),(5,-1)])
    (5.0, -0.0)
    """
    p = line_intersection(line1, line2)
    if p is None:
        return None

    is_on_line1 = is_point_on_line_segment(line1, p)
    is_on_line2 = is_point_on_line_segment(line2, p)
    if is_on_line1 and is_on_line2:
        return p

    return None


def is_point_in_polygon(poly, point):
    """
    accepted types of poly: matrix, ndarray, list of tuples
    @type point: tuple
    """
    if type(poly) is np.matrix:
        poly = poly.A
    elif type(poly) is list:
        poly = np.array(poly)

    polygon = mpl_path.Path(poly)
    return polygon.contains_point(point)


def find_overlapping_polygon(poly1, poly2):
    """
    Input values are n by 2 matrices containing n points
    @type poly1: ndarray
    @type poly2: ndarray
    @rtype: ndarray
    """
    if not np.array_equal(poly1[0], poly1[-1]):
        raise Exception("poly1 is not enclosed properly. poly1[0] must equal poly1[-1]")
    if not np.array_equal(poly2[0], poly2[-1]):
        raise Exception("poly2 is not enclosed properly. poly2[0] must equal poly2[-1]")

    points = []

    if type(poly1) == np.matrix:
        poly1 = poly1.A

    if type(poly2) == np.matrix:
        poly2 = poly2.A

    # find intersection points between two polygon paths
    for i in range(len(poly1) - 1):
        line_segment1 = map(tuple, poly1[i:i + 2])
        for j in range(len(poly2) - 1):
            line_segment2 = map(tuple, poly2[j:j + 2])
            p = line_segment_intersection(line_segment1, line_segment2)
            if p is not None:
                points.append(p)  # found an intersection point

    # find vertices contained in another polygon
    for p in poly1[:-1]:
        p = tuple(p)
        if is_point_in_polygon(poly2, p):
            points.append(p)

    for p in poly2[:-1]:
        p = tuple(p)
        if is_point_in_polygon(poly1, p):
            points.append(p)

    if len(points) == 0:
        return None

    # reconstruct the polygon from unordered points
    # Reference: http://stackoverflow.com/questions/10846431/
    #   ordering-shuffled-points-that-can-be-joined-to-form-a-polygon-in-python
    # compute centroid
    cent = (sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points))
    # remove duplicates
    points = list(set(points))
    # sort by polar angle
    points.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
    # enclose the polygon
    points += [points[0]]

    if find_polygon_area(points) > 0:
        return np.array(points)
    return None


def find_overlapping_polygon_area(poly1, poly2):
    """
    @type poly1: ndarray
    @type poly2: ndarray
    @rtype: float
    """
    poly = find_overlapping_polygon(poly1, poly2)
    if poly is None:
        return None
    return find_polygon_area(poly)


def is_point_on_line_segment(line, point, epsilon=0.05):
    """
    @type line: list
    @type point: tuple
    """
    cross_product = np.cross(np.array(line[1]) - np.array(line[0]),
                             np.array(point) - np.array(line[0]))
    are_aligned = -epsilon < cross_product < epsilon
    x_points = [line[0][0], line[1][0]]
    y_points = [line[0][1], line[1][1]]
    x_points.sort()
    y_points.sort()
    is_x_in_range = x_points[0] <= point[0] <= x_points[1]
    is_y_in_range = y_points[0] <= point[1] <= y_points[1]

    return are_aligned and is_x_in_range and is_y_in_range


def check_polygon_equality(poly1, poly2, places=4):
    """
    Assume the polygon does not contain duplicate points.
    @type poly1: ndarray
    @type poly2: ndarray
    @type places: int
    @rtype: bool
    """
    if not np.array_equal(poly1[0], poly1[-1]):
        raise Exception("poly1 is not enclosed properly. poly1[0] must equal poly1[-1]")
    if not np.array_equal(poly2[0], poly2[-1]):
        raise Exception("poly2 is not enclosed properly. poly2[0] must equal poly2[-1]")

    poly1_rounded = np.copy(poly1).round(places)[:-1]
    poly2_rounded = np.copy(poly2).round(places)[:-1]
    val, min_idx1 = min(((val[0], val[1]), idx) for (idx, val) in enumerate(poly1_rounded))
    val2, min_idx2 = min(((val[0], val[1]), idx) for (idx, val) in enumerate(poly2_rounded))
    rotated1 = np.roll(poly1_rounded, -min_idx1, 0)
    rotated2 = np.roll(poly2_rounded, -min_idx2, 0)
    if np.array_equal(rotated1, rotated2):
        return True
    return np.array_equal(rotated1, np.roll(rotated2[::-1, :], 1, 0))


def rotate_rects(rects, center, angle):
    """
    @param rects: n by 5 by 2 array of n rectangles.
        Each rectangle consists of five (x, y) coordinates. Any polygon would work, in fact.
    @type rects: ndarray
    @type center: tuple
    @type angle: float
    @rtype: ndarray
    """
    # 2 by 3 rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    n, p, d = rects.shape

    # 2 by n*5 array
    points = vstack((rects.T.reshape(d, n * p, order='F'), ones((1, n * p))))

    rotated_points = np.dot(rot_mat, points)
    rotated_rects = rotated_points.T.reshape(n, p, d)

    return rotated_rects


def rect_to_polygon(rects):
    """
    Convert rectangles to polygon format. Each rectangle will be represented as five (x, y) vertices.
    @param rects: n by 4 array of n rectangles where each row is (x, y, w, h)
    @returns: n by 5 by 2 array of n polygons
    @type rects: ndarray
    @rtype: ndarray
    """
    # tile the x, y points
    rects = np.array([rects])
    poly = np.tile(np.transpose(rects[:, :, 0:2], (1, 0, 2)), (1, 5, 1))

    # width and height
    wh = np.transpose(rects[:, :, 2:4], (1, 0, 2)).reshape(poly.shape[0], 2)
    poly[:, 1, 0] += wh[:, 0]
    poly[:, 2, :] += wh[:, :]
    poly[:, 3, 1] += wh[:, 1]

    return poly
