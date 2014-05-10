import cv2
import numpy as np

__author__ = 'Daeyun Shin'


def rotate_image(image, angle=90):
    """
    Rotate a given image without cropping
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


def crop_images(image, rects):
    """
    @type image: ndarray
    @type rects: list
    """
    results = []
    for rect in rects:
        x, y, w, h = rect
        roi = image[y:y + h + 1, x:x + w + 1]
        results.append(roi)

    return results

def crop_image(image, roi_rect):
    """
    @type image: ndarray
    @type rects: list
    """
    x, y, w, h = roi_rect
    cropped_img = image[y:y + h + 1, x:x + w + 1]
    return cropped_img


if __name__ == "__main__":
    import doctest

    doctest.testmod()
