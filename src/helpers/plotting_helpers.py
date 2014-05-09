import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

__author__ = 'Daeyun Shin'


def plot_rects_on_image(image, rects_list, colors=None):
    """
    @param rects: list of ndarrays
    @param colors: list of color of the corresponding set of rectangles.
                   For example, ['#ff0000', '#0000ff']
    @type image: ndarray
    @type rects: list
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0,  image.shape[0])
    ax.set_ylim(ax.get_ylim()[::-1])

    print rects_list

    for ind, rects in enumerate(rects_list):
        try:
            color = colors[ind]
        except:
            color = 'black'

        if rects.ndim == 2:
            rects = [rects]

        for rect in rects:
            add_polygon_to_axis(ax, rect, color)

    plt.imshow(image, origin='lower')
    plt.show()


def add_polygon_to_axis(axis, poly, color='black'):
    path = Path(poly)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=2)
    axis.add_patch(patch)
