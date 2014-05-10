import os
import uuid
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys
from helpers.config_helpers import parse_config
from helpers.io_helpers import get_absolute_path, make_sure_dir_exists, ensure_extension

__author__ = 'Daeyun Shin'


def add_polygon_to_axis(axis, poly, color='black'):
    path = Path(poly)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=2)
    axis.add_patch(patch)


def save_figure(fig, filename=None, path=None, ext='jpg'):
    """
    save a matplotlib
    """
    if path is None:
        config_path = '../../config.yaml'
        config = parse_config(get_absolute_path(config_path))
        fig_save_dir = config['paths']['output']['figure save dir']
    else:
        fig_save_dir = path

    make_sure_dir_exists(fig_save_dir)

    if filename is None:
        # default option: randomly generate a filename
        full_filename = os.path.join(fig_save_dir, uuid.uuid4().hex[:8])
    else:
        full_filename = os.path.join(fig_save_dir, filename)

    full_filename = ensure_extension(full_filename, ext)
    fig.savefig(full_filename)
    print "saved figure {}".format(full_filename)


def plot_polygons_on_image(image, rects_list, colors=None, is_RGB_flipped=True, will_save_to_file=False,
                           filename=None, will_block=True, click_to_save=True):
    """
    @param rects: list of ndarrays
    @param colors: list of color of the corresponding set of rectangles.
                   For example, ['#ff0000', '#0000ff']
    @type image: ndarray
    @type filename: string
    @type rects: list
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    if is_RGB_flipped:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
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

    ax.imshow(image, origin='lower')

    if will_save_to_file:
        save_figure(fig, filename=filename, path=None, ext='jpg')

    if click_to_save:
        def press(event):
            save_figure(fig, filename=filename, path=None, ext='jpg')
        fig.canvas.mpl_connect('button_press_event', press)

    if will_block:
        plt.show()
    else:
        fig.show()
