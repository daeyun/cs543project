import cv2
import numpy as np
import matplotlib.path as mpl_path

__author__ = 'Daeyun Shin'


def find_polygon_area(array):
    """
    >>> find_polygon_area([(0, 0), (10, 0), (10, 10), (0, 10),  (0, 0)])
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
    g = np.linalg.det([[a, c],[b, d]])
    h = np.linalg.det([[c, e],[d, f]])
    i = np.linalg.det([[a, e],[b, f]])

    if h == 0:
        # lines are parallel
        return None

    x, y = g/h, i/h

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
    is_on_line1 = is_point_on_line_segment(line1, p)
    is_on_line2 = is_point_on_line_segment(line2, p)
    if p is not None and is_on_line1 and is_on_line2:
        return p
    return None


def is_point_in_polygon(poly, point):
    pass


def get_polygon_overlapping_area(poly1, poly2):
    pass


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
