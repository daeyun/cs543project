import os
from helpers.io_helpers import path_to_filename, make_sure_dir_exists, save_image

__author__ = 'Daeyun Shin'


def save_image_patch(image_patch, patch_info):
    """
    @type image_patch: ndarray
    @type patch_info: dict

    patch_info = {
        'patch': { 'x': x, 'y': y, 'w': w, 'h': h, 'label': label, },
        'source': { 'theta': theta, 'path': image_path, },
        'out dir': out_dir
    }
    """
    image_filename = path_to_filename(patch_info['source']['path'])

    name, ext = image_filename.rsplit('.', 1)
    if 'dtheta' not in patch_info['source']:
        patch_info['source']['dtheta'] = 0

    out_name = "{name}.{ext}__{x}_{y}_{w}_{h}_{theta}_{dtheta}.{ext}".format(
        name=name,
        ext=ext,
        x=patch_info['patch']['x'],
        y=patch_info['patch']['y'],
        w=patch_info['patch']['w'],
        h=patch_info['patch']['h'],
        theta=patch_info['source']['theta'],
        dtheta=patch_info['source']['dtheta']
    )
    out_dir = patch_info['out dir']
    make_sure_dir_exists(out_dir)
    out_path = os.path.join(out_dir, out_name)
    print out_path
    save_image(out_path, image_patch)