import os
from helpers.io_helpers import path_to_filename, save_image


def save_image_patch(image_patch, patch_info):
    """
    @type image_patch: ndarray
    @type patch_info: dict

    patch_info = {
        'patch': { 'x': x, 'y': y, 'w': w, 'h': h, 'label': label, },
        'source': { 'orientation': orientation, 'path': image_path, },
        'out dir': out_dir
    }
    """
    image_filename = path_to_filename(patch_info['source']['path'])

    name, ext = image_filename.rsplit('.', 1)
    out_name = "{name}.{x}.{y}.{w}.{h}.{orientation}.{ext}".format(
        name=name,
        ext=ext,
        x=patch_info['patch']['x'],
        y=patch_info['patch']['y'],
        w=patch_info['patch']['w'],
        h=patch_info['patch']['h'],
        orientation=patch_info['source']['orientation']
    )
    out_path = os.path.join(patch_info['out dir'], out_name)
    save_image(out_path, image_patch)
