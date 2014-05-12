from helpers.feature_extraction_helpers import get_intersecting_rect2
from helpers.image_manager import ImageManager
from helpers.plotting_helpers import plot_rects_on_image

__author__ = 'Daeyun Shin'

# load all annotation files in the given directory
image_manager = ImageManager('./img')

# raises exception if annotation does not exist for this image
image, rect_sets = image_manager.load_annotated_image('./img/DSCF0876.JPG')

base_rect = (500, 500, 130, 130)
for i in range(300, 1000, 40):
    for j in range(300, 1200, 40):
        rect = (i, j, 290, 280)
        irect = get_intersecting_rect2(base_rect, rect)
        if irect is not None:
            print irect
            plot_rects_on_image(image, [base_rect, rect, irect], ['yellow', 'green', 'magenta'])
        else:
            print "no intersection"
